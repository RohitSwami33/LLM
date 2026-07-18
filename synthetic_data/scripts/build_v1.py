#!/usr/bin/env python3
"""Build the v1 instruction dataset end-to-end (offline).

Pipeline:
    1. generate   -> per-task JSONL via the offline teacher (verified math /
                      code / sql, templated text tasks)
    2. filter     -> quality + dedup pipeline
    3. score      -> reasoning/confidence scorers + composite quality_score
    4. export     -> normalized records to JSONL / Parquet / Arrow / HF

Run:
    python -m synthetic_data.scripts.build_v1 -c configs/instruction_v1.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from synthetic_data.core.config import load_config  # noqa: E402
from synthetic_data.core.telemetry import Metrics, setup_logger  # noqa: E402
from synthetic_data.core.schema import Sample, loads, dumps  # noqa: E402
from synthetic_data.generator.teacher import build_teacher  # noqa: E402
from synthetic_data.generator.batching import BatchGenerator  # noqa: E402
from synthetic_data.filtering import build_filters  # noqa: E402
from synthetic_data.scoring import build_scorers  # noqa: E402
from synthetic_data.exporters import instruction as instexp  # noqa: E402

logger = setup_logger("build_v1")

OUT_ROOT = Path("datasets/synthetic/v1")


def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield loads(line)


def _quality_score(sample: Sample, reasoning_val: float) -> float:
    """Composite offline quality proxy in [0, 1]."""
    if sample.reasoning:
        # reasoning present -> trust the reasoning scorer signal.
        return round(max(0.0, min(1.0, reasoning_val)), 4)
    # No explicit reasoning -> score by completeness / length.
    resp = sample.response or ""
    score = min(1.0, len(resp) / 250.0) * 0.85
    if sample.metadata.get("input"):
        score += 0.1
    if 40 <= len(resp) <= 4000:
        score += 0.05
    return round(max(0.0, min(1.0, score)), 4)


async def run(config_path: str):
    config = load_config(config_path)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    gen_dir = Path(config.generation.output_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. generate ------------------------------------------------------
    teacher = build_teacher(config)
    gen_metrics = Metrics()
    for task in config.tasks:
        if not task.enabled:
            continue
        logger.info("GENERATE %s (%d samples)", task.task_type, task.num_samples)
        gen = BatchGenerator(
            task=task, teacher=teacher, gen_config=config.generation,
            teacher_config=config.teacher, metrics=gen_metrics,
            output_dir=gen_dir, seed=config.generation.seed,
        )
        await gen.run()
    gen_metrics.log(logger)

    generated = list(gen_dir.glob("*.jsonl"))
    logger.info("Generated files: %d", len(generated))

    # ---- 2. filter --------------------------------------------------------
    filt = build_filters(config.filters)
    filt_metrics = Metrics()
    accepted: list[Sample] = []
    rejected = Counter()
    out_filt = OUT_ROOT / "filtered.jsonl"
    with open(out_filt, "w", encoding="utf-8") as fout:
        for gp in generated:
            for sample in _read_jsonl(gp):
                filt_metrics.attempted += 1
                keep, reason = filt.check(sample)
                if keep:
                    filt_metrics.accepted += 1
                    accepted.append(sample)
                    fout.write(dumps(sample) + "\n")
                else:
                    filt_metrics.rejected += 1
                    rejected[reason or "unknown"] += 1
    logger.info("FILTER accepted=%d rejected=%d", filt_metrics.accepted, filt_metrics.rejected)
    for r, c in rejected.most_common():
        logger.info("  rejected[%s] = %d", r, c)

    # ---- 3. score ---------------------------------------------------------
    scorer = build_scorers(config.scoring)
    scored: list[Sample] = []
    out_scored = OUT_ROOT / "scored.jsonl"
    scored_metrics = Metrics()
    if scorer is not None:
        async def _score_one(s: Sample):
            scores = await scorer.score(s)
            rv = scores.get("reasoning", 0.0) or 0.0
            s.metadata["scores"] = scores
            s.metadata["quality_score"] = _quality_score(s, rv)
            return s

        tasks = [_score_one(s) for s in accepted]
        chunk = 2000
        with open(out_scored, "w", encoding="utf-8") as fout:
            for i in range(0, len(tasks), chunk):
                done = await asyncio.gather(*tasks[i:i + chunk])
                for s in done:
                    scored_metrics.scored += 1
                    scored.append(s)
                    fout.write(dumps(s) + "\n")
        logger.info("SCORE scored=%d", scored_metrics.scored)
    else:
        scored = accepted
        with open(out_scored, "w", encoding="utf-8") as fout:
            for s in accepted:
                s.metadata["quality_score"] = _quality_score(s, 0.0)
                scored.append(s)
                fout.write(dumps(s) + "\n")

    # ---- 4. export --------------------------------------------------------
    exp = config.export
    base = Path(exp.output_dir)
    base.mkdir(parents=True, exist_ok=True)

    n_jsonl = instexp.export(scored, "jsonl", base / "dataset.jsonl")
    n_parquet = instexp.export(scored, "parquet", base / "dataset.parquet")
    n_arrow = instexp.export(scored, "arrow", base / "dataset.arrow")
    n_hf = instexp.export(
        scored, "hf", base / "hf_dataset",
        push_to_hub=exp.push_to_hub, dataset_name=exp.dataset_name,
    )
    logger.info("EXPORT jsonl=%d parquet=%d arrow=%d hf=%d",
                n_jsonl, n_parquet, n_arrow, n_hf)

    # ---- stats --------------------------------------------------------------
    task_counts = Counter(s.task_type for s in scored)
    diff_counts = Counter(s.metadata.get("difficulty", "medium") for s in scored)
    lang_counts = Counter(s.metadata.get("language", "en") for s in scored)
    q_scores = [s.metadata.get("quality_score", 0.0) for s in scored]
    avg_q = sum(q_scores) / len(q_scores) if q_scores else 0.0

    stats = {
        "total_samples": len(scored),
        "per_task": dict(task_counts),
        "per_difficulty": dict(diff_counts),
        "per_language": dict(lang_counts),
        "avg_quality_score": round(avg_q, 4),
        "generated_files": [str(p) for p in generated],
        "outputs": {
            "jsonl": str(base / "dataset.jsonl"),
            "parquet": str(base / "dataset.parquet"),
            "arrow": str(base / "dataset.arrow"),
            "hf": str(base / "hf_dataset"),
        },
        "filters": {"accepted": filt_metrics.accepted, "rejected": filt_metrics.rejected,
                     "reasons": dict(rejected)},
    }
    (OUT_ROOT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    logger.info("STATS total=%d avg_quality=%.3f", len(scored), avg_q)
    logger.info("STATS per_task=%s", dict(task_counts))
    logger.info("STATS per_difficulty=%s", dict(diff_counts))
    logger.info("Done -> %s", OUT_ROOT)


def main():
    ap = argparse.ArgumentParser(description="Build v1 instruction dataset (offline)")
    ap.add_argument("-c", "--config", default="configs/instruction_v1.yaml")
    args = ap.parse_args()
    cfg = Path(args.config)
    if not cfg.exists() and (ROOT / args.config).exists():
        cfg = ROOT / args.config
    asyncio.run(run(str(cfg)))


if __name__ == "__main__":
    main()
