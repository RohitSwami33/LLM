#!/usr/bin/env python3
"""Generate synthetic data for the enabled tasks.

Reads a YAML config, instantiates the teacher, and runs
:class:`~synthetic_data.generator.batching.BatchGenerator` per task.  Progress,
token usage and estimated cost are logged continuously.

Examples
--------
    python -m synthetic_data.scripts.generate -c configs/generation.yaml
    python -m synthetic_data.scripts.generate --task reasoning --num 500
    python -m synthetic_data.scripts.generate --no-resume
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make the repository root importable when run as a script.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from synthetic_data.core.config import load_config  # noqa: E402
from synthetic_data.core.telemetry import Metrics, setup_logger  # noqa: E402
from synthetic_data.generator.teacher import build_teacher  # noqa: E402
from synthetic_data.generator.batching import BatchGenerator  # noqa: E402

logger = setup_logger("generate")


async def run(config_path: str, task: str | None, num: int | None, resume: bool):
    config = load_config(config_path)
    if not resume:
        config.generation.resume = False
    if task:
        # Restrict to a single task type (overriding enabled flags).
        for t in config.tasks:
            t.enabled = t.task_type == task
        if num is not None:
            for t in config.tasks:
                if t.task_type == task:
                    t.num_samples = num

    metrics = Metrics()
    teacher = build_teacher(config)
    out_dir = Path(config.generation.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enabled = [t for t in config.tasks if t.enabled]
    if not enabled:
        logger.warning("No enabled tasks found in config.")
        return

    logger.info("Generating %d task(s) → %s", len(enabled), out_dir)
    for task_cfg in enabled:
        gen = BatchGenerator(
            task=task_cfg,
            teacher=teacher,
            gen_config=config.generation,
            teacher_config=config.teacher,
            metrics=metrics,
            output_dir=out_dir,
            seed=config.generation.seed,
        )
        logger.info("Starting task '%s' (%d samples)", task_cfg.task_type, task_cfg.num_samples)
        await gen.run()
        metrics.log(logger)

    metrics.log(logger)
    logger.info("Done. Output directory: %s", out_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("-c", "--config", default="configs/generation.yaml")
    parser.add_argument("--task", default=None, help="Restrict to one task_type")
    parser.add_argument("--num", type=int, default=None, help="Override num_samples for --task")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints")
    parser.add_argument("--base-dir", default=None, help="Override config output base dir")
    args = parser.parse_args()

    # Allow running from repo root or synthetic_data dir.
    cfg_path = Path(args.config)
    if not cfg_path.exists() and (ROOT / args.config).exists():
        cfg_path = ROOT / args.config

    asyncio.run(run(str(cfg_path), args.task, args.num, not args.no_resume))


if __name__ == "__main__":
    main()
