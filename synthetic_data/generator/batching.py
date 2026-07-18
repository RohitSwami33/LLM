"""Async batch generation with checkpointing and resume support.

:class:`BatchGenerator` drives the teacher to produce many synthetic samples
for a given task type:

* bounded concurrency (semaphore sized by ``batch_size``),
* RPM-aware rate limiting,
* automatic retries (in :class:`~synthetic_data.generator.teacher.DeepSeekTeacher`),
* incremental JSONL writes (so an interrupted run can resume),
* live progress bars and metrics.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional, Set

from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm

from ..core.config import GenerationConfig, TeacherConfig, TaskConfig
from ..core.schema import Sample, dumps
from ..core.telemetry import Metrics
from ..core.rate_limit import RateLimiter
from .teacher import BaseTeacher
from .templates import get_template
from .prompts import SeedProvider, render, parse_response, build_sample


def _load_existing_ids(path: Path) -> Set[str]:
    """Recover already-generated sample ids from a JSONL checkpoint file."""
    ids: Set[str] = set()
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ids.add(obj.get("id"))
            except json.JSONDecodeError:
                continue
    return ids


class BatchGenerator:
    """Generate one task type's samples concurrently and durably."""

    def __init__(
        self,
        task: TaskConfig,
        teacher: BaseTeacher,
        gen_config: GenerationConfig,
        teacher_config: TeacherConfig,
        metrics: Metrics,
        output_dir: Path,
        seed: int = 42,
    ):
        self.task = task
        self.teacher = teacher
        self.gen = gen_config
        self.teacher_config = teacher_config
        self.metrics = metrics
        self.output_dir = output_dir
        self.template = get_template(task.template)
        self.seeds = SeedProvider(seed=seed)
        self.rate_limiter = RateLimiter(teacher_config.rpm_limit)
        self._sem = asyncio.Semaphore(max(1, gen_config.batch_size))

    # ------------------------------------------------------------------ #
    async def _worker(self, idx: int) -> Optional[Sample]:
        seed = self.seeds.get(self.task.task_type, idx)
        seed.total = self.task.num_samples  # used by the offline teacher for difficulty split
        system, user = render(self.template, seed)
        self.metrics.attempted += 1
        try:
            result = await self.teacher.generate(
                user,
                system=system,
                temperature=self.task.temperature,
                top_p=self.task.top_p,
                max_tokens=self.task.max_tokens,
                template=self.template,
                seed=seed,
            )
        except Exception as exc:  # network / API failure after retries
            self.metrics.failed += 1
            try:
                with open("/tmp/gen_errors.log", "a", encoding="utf-8") as _ef:
                    if self.metrics.failed <= 20:
                        import traceback as _tb
                        _ef.write(f"IDX={idx} TASK={self.task.task_type} :: {type(exc).__name__}: {exc}\n")
                        _tb.print_exc(file=_ef)
            except Exception:
                pass
            return None
        except Exception as exc:  # network / API failure after retries
            self.metrics.failed += 1
            return None

        self.metrics.completed += 1
        self.metrics.prompt_tokens += result.prompt_tokens
        self.metrics.completion_tokens += result.completion_tokens
        self.metrics.cost_usd += result.cost_usd

        try:
            fields = parse_response(self.template, result.text)
            return build_sample(self.task.task_type, self.template, seed, fields, result)
        except Exception as exc:
            self.metrics.failed += 1
            return None

    async def run(self) -> Path:
        """Run generation, writing results to a per-task JSONL file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"{self.task.task_type}.jsonl"

        done = _load_existing_ids(out_path) if self.gen.resume else set()
        total = self.task.num_samples
        pending = [i for i in range(total) if f"{self.task.task_type}-{i}" not in done]

        if not pending:
            self.metrics.written += total
            return out_path

        self.metrics.written = len(done)
        pbar = atqdm(
            total=len(pending),
            desc=f"gen:{self.task.task_type}",
            unit="ex",
        )

        # Open in append mode for incremental, crash-safe writes.
        with open(out_path, "a", encoding="utf-8") as fout:
            # Process in chunks to bound memory for millions of samples.
            for start in range(0, len(pending), self.gen.batch_size):
                chunk = pending[start : start + self.gen.batch_size]
                tasks = [self._spawn(i, fout) for i in chunk]
                for coro in asyncio.as_completed(tasks):
                    sample = await coro
                    if sample is not None:
                        fout.write(dumps(sample) + "\n")
                        fout.flush()
                        self.metrics.written += 1
                    pbar.update(1)
        pbar.close()
        return out_path

    async def _spawn(self, idx: int, fout) -> Optional[Sample]:
        async with self._sem:
            await self.rate_limiter.acquire()
            return await self._worker(idx)
