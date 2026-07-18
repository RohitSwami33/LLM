#!/usr/bin/env python3
"""
SMALL Mixture Training Pipeline
================================
Fully automated end-to-end pipeline:
  1. Monitor SMALL mixture build completion
  2. Train SentencePiece tokenizer (vocab=32768)
  3. Validate training pipeline (fwd/bwd/step/ckpt)
  4. Launch 1B-token pretraining with auto-resume
  5. Run evaluation suite after training
  6. Update leaderboard + generate reports

Usage:
    python scripts/run_small_pipeline.py
    python scripts/run_small_pipeline.py --skip-build    # assume build is done
    python scripts/run_small_pipeline.py --eval-only     # only run eval on existing experiment
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIXTURE_DIR = PROJECT_ROOT / "datasets" / "mixtures" / "small"
CORPUS_PATH = MIXTURE_DIR / "corpus.jsonl"
TOKENIZER_DATA_DIR = MIXTURE_DIR / "tokenizer_data"
METADATA_PATH = MIXTURE_DIR / "metadata.json"

TOKENIZER_MODEL_SRC = PROJECT_ROOT / "training" / "tokenizer" / "small_tokenizer.model"
TOKENIZER_MODEL_DST = PROJECT_ROOT / "training" / "tokenizer" / "small_tokenizer.model"

TRAINING_CONFIG = PROJECT_ROOT / "training" / "configs" / "pretrain_small.yaml"
TRAIN_SCRIPT = PROJECT_ROOT / "training" / "scripts" / "train.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate.py"
LEADERBOARD_SCRIPT = PROJECT_ROOT / "scripts" / "leaderboard.py"  # may not exist
COMPARE_SCRIPT = PROJECT_ROOT / "scripts" / "compare.py"

CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

PIPELINE_STATE = PROJECT_ROOT / "pipeline_state.json"

VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

LOG = logging.getLogger("pipeline")
LOG_FMT = "%(asctime)s [%(levelname)-7s] %(message)s"

# Expected corpus size (approximate)
EXPECTED_MIN_DOCS = 3_000_000
EXPECTED_MAX_DOCS = 4_000_000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    LOG.setLevel(logging.DEBUG)
    if not LOG.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(LOG_FMT))
        LOG.addHandler(ch)
        fh = logging.FileHandler(PROJECT_ROOT / "pipeline.log", mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(LOG_FMT))
        LOG.addHandler(fh)

# ---------------------------------------------------------------------------
# Pipeline state (for resumability)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if PIPELINE_STATE.exists():
        with open(PIPELINE_STATE) as f:
            return json.load(f)
    return {"phase": "init", "started": datetime.now(timezone.utc).isoformat()}

def save_state(state: dict):
    with open(PIPELINE_STATE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def set_phase(state: dict, phase: str):
    state["phase"] = phase
    state[f"{phase}_started"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    LOG.info("=== Phase: %s ===", phase)

# ---------------------------------------------------------------------------
# 1. Monitor build completion
# ---------------------------------------------------------------------------

def wait_for_build(timeout: int = 7200) -> bool:
    """Wait for metadata.json to appear and build process to exit."""
    LOG.info("Waiting for SMALL mixture build to complete...")
    LOG.info("  Expected outputs: %s", CORPUS_PATH)
    LOG.info("  Timeout: %d seconds", timeout)

    start = time.time()
    build_pid = _find_build_pid()
    if build_pid:
        LOG.info("  Found build process PID: %d", build_pid)
    else:
        LOG.info("  No build process found — build may already be done")

    while time.time() - start < timeout:
        # Check if metadata.json exists (final output of build)
        if METADATA_PATH.exists():
            # Give it a few seconds for file to flush
            time.sleep(2)
            # Verify build process has exited
            if build_pid and _is_pid_running(build_pid):
                LOG.info("  metadata.json exists but build process still running, waiting...")
                time.sleep(5)
                continue
            LOG.info("  Build completed!")
            return True

        # Check if build process died
        if build_pid and not _is_pid_running(build_pid):
            LOG.warning("  Build process (PID %d) exited without creating metadata.json", build_pid)
            # Check if build output log has errors
            log_path = PROJECT_ROOT / "build_small.log"
            if log_path.exists():
                with open(log_path) as f:
                    lines = f.readlines()
                    error_lines = [l for l in lines[-20:] if "error" in l.lower() or "traceback" in l.lower()]
                    if error_lines:
                        LOG.error("Build errors:\n%s", "".join(error_lines))
            return False

        elapsed = int(time.time() - start)
        if elapsed % 60 == 0 and elapsed > 0:
            LOG.info("  Waiting... (%ds elapsed)", elapsed)
        time.sleep(5)

    LOG.error("Timeout waiting for build after %ds", timeout)
    return False


def _find_build_pid() -> Optional[int]:
    """Find PID of build_mixture.py process."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "build_mixture.py.*small"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def verify_build() -> dict:
    """Verify build outputs are valid."""
    LOG.info("Verifying build outputs...")
    errors = []

    # Check corpus.jsonl
    if not CORPUS_PATH.exists():
        errors.append(f"corpus.jsonl not found: {CORPUS_PATH}")
    else:
        size = CORPUS_PATH.stat().st_size
        LOG.info("  corpus.jsonl: %.1f MB", size / 1e6)
        if size == 0:
            errors.append("corpus.jsonl is empty")

    # Check tokenizer_data/
    if not TOKENIZER_DATA_DIR.exists():
        errors.append(f"tokenizer_data/ not found: {TOKENIZER_DATA_DIR}")
    else:
        txt_files = list(TOKENIZER_DATA_DIR.glob("*.txt"))
        LOG.info("  tokenizer_data/: %d text files", len(txt_files))
        if len(txt_files) == 0:
            errors.append("tokenizer_data/ contains no .txt files")

    # Check metadata.json
    metadata = None
    if not METADATA_PATH.exists():
        errors.append(f"metadata.json not found: {METADATA_PATH}")
    else:
        try:
            with open(METADATA_PATH) as f:
                metadata = json.load(f)
            n_docs = metadata.get("total_documents", 0)
            est_tokens = metadata.get("estimated_tokens", 0)
            LOG.info("  metadata.json: %d docs, ~%s estimated tokens",
                     n_docs, f"{est_tokens:,}")
            if n_docs < EXPECTED_MIN_DOCS:
                errors.append(f"Too few documents: {n_docs} < {EXPECTED_MIN_DOCS}")
        except json.JSONDecodeError as e:
            errors.append(f"metadata.json is corrupted: {e}")

    if errors:
        for e in errors:
            LOG.error("BUILD VERIFICATION FAILED: %s", e)
        raise RuntimeError(f"Build verification failed with {len(errors)} errors")

    LOG.info("Build verification passed")
    return metadata or {}

# ---------------------------------------------------------------------------
# 2. Train tokenizer
# ---------------------------------------------------------------------------

def train_tokenizer() -> dict:
    """Train SentencePiece tokenizer on corpus. Returns stats."""
    LOG.info("Training SentencePiece tokenizer (vocab=32768)...")

    if TOKENIZER_MODEL_SRC.exists():
        LOG.info("  Tokenizer already exists at %s, retraining", TOKENIZER_MODEL_SRC)

    # Concatenate all tokenizer text shards into a single file for SentencePiece
    corpus_for_sp = PROJECT_ROOT / "datasets" / "mixtures" / "small" / "tokenizer_corpus.txt"
    if not corpus_for_sp.exists():
        LOG.info("  Concatenating tokenizer shards...")
        with open(corpus_for_sp, "w", encoding="utf-8") as out:
            for shard in sorted(TOKENIZER_DATA_DIR.glob("*.txt")):
                with open(shard, encoding="utf-8") as inp:
                    shutil.copyfileobj(inp, out)
        LOG.info("  Written %s", corpus_for_sp)

    # Train tokenizer
    result = subprocess.run(
        [
            str(VENV_PYTHON), str(PROJECT_ROOT / "training" / "scripts" / "train_tokenizer.py"),
            "--data", str(corpus_for_sp),
            "--output", str(TOKENIZER_MODEL_SRC),
            "--vocab-size", "32768",
            "--type", "sentencepiece",
        ],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        LOG.error("Tokenizer training failed:\n%s", result.stderr)
        raise RuntimeError("Tokenizer training failed")
    LOG.info("  %s", result.stdout.strip())

    # Collect tokenizer stats
    stats = _compute_tokenizer_stats(corpus_for_sp)
    LOG.info("  Tokenizer stats:")
    for k, v in stats.items():
        LOG.info("    %s: %s", k, v)
    return stats


def _compute_tokenizer_stats(corpus_path: Path) -> dict:
    """Compute tokenizer statistics."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(TOKENIZER_MODEL_SRC))

    total_tokens = 0
    total_docs = 0
    total_unknown = 0
    total_chars = 0

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = sp.EncodeAsIds(line)
            total_tokens += len(ids)
            total_unknown += sum(1 for x in ids if x == sp.unk_id())
            total_chars += len(line)
            total_docs += 1

            # Sample for speed
            if total_docs >= 10000:
                break

    avg_tokens = total_tokens / max(total_docs, 1)
    compression_ratio = total_chars / max(total_tokens, 1)
    unknown_rate = total_unknown / max(total_tokens, 1)

    return {
        "vocab_size": sp.GetPieceSize(),
        "sampled_docs": total_docs,
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": round(avg_tokens, 2),
        "compression_ratio": round(compression_ratio, 2),
        "unknown_token_rate": round(unknown_rate, 6),
        "vocabulary_coverage": round(1 - unknown_rate, 6),
    }


def copy_tokenizer():
    """Copy tokenizer to training directory."""
    dst = TOKENIZER_MODEL_DST
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TOKENIZER_MODEL_SRC, dst)
    LOG.info("Tokenizer copied to %s", dst)

# ---------------------------------------------------------------------------
# 3. Validate training pipeline
# ---------------------------------------------------------------------------

def validate_pipeline() -> bool:
    """Run one fwd/bwd/step/ckpt/eval cycle to validate pipeline."""
    LOG.info("Validating training pipeline (fwd/bwd/step/ckpt/load/eval)...")

    code = textwrap.dedent("""\
import sys, os, torch, json, time
sys.path.insert(0, os.getcwd())

# 1. Load config
import yaml
with open("training/configs/pretrain_small.yaml") as f:
    config = yaml.safe_load(f)

# 2. Build components
from training.trainer import Trainer
trainer = Trainer(config)
print(f"  Model: {sum(p.numel() for p in trainer.model.parameters())/1e6:.1f}M params")
print(f"  Device: {trainer.device}")

# 3. One forward + backward pass
batch = next(iter(trainer.dataloader))
metrics = trainer.train_step(batch)
print(f"  Forward+backward OK | loss={metrics['loss']:.4f}")

# 4. One optimizer step
trainer.global_step += 1

# 5. Save checkpoint
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, "test_ckpt")
    from training.utils.checkpoint import save_checkpoint, load_checkpoint
    save_checkpoint(
        path=ckpt_path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        ema=trainer.ema,
        step=trainer.global_step,
        epoch=trainer.epoch,
        total_tokens=trainer.total_tokens,
    )
    print(f"  Checkpoint save OK")

    # 6. Load checkpoint
    from copy import deepcopy
    model_copy = type(trainer.model)(trainer.model.model_config)
    model_copy.load_state_dict(deepcopy(trainer.model.state_dict()))
    state = load_checkpoint(ckpt_path, model_copy, trainer.optimizer,
                           trainer.scheduler, trainer.scaler, trainer.ema, trainer.device)
    print(f"  Checkpoint load OK | step={state['step']}, tokens={state['total_tokens']}")

# 7. One validation step
val_results = None
try:
    eval_steps = config.get("training", {}).get("eval_steps", 5)
    from training.trainer import evaluate
    val_results = evaluate(
        trainer.model, trainer.val_dataloader, trainer.device,
        max_steps=eval_steps, pad_token_id=trainer.tokenizer.pad_token_id,
    )
    print(f"  Validation OK | val_loss={val_results['val_loss']:.4f}, val_ppl={val_results['val_perplexity']:.2f}")
except Exception as e:
    print(f"  Validation failed: {e}")
    sys.exit(1)

print("PIPELINE VALIDATION PASSED")
""")
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", code],
        capture_output=True, text=True, timeout=120,
        cwd=str(PROJECT_ROOT)
    )
    LOG.info("Validation output:\n%s", result.stdout)
    if result.returncode != 0:
        LOG.error("Validation failed:\n%s", result.stderr)
        raise RuntimeError("Pipeline validation failed")
    if "PIPELINE VALIDATION PASSED" not in result.stdout:
        raise RuntimeError("Pipeline validation incomplete")
    LOG.info("Pipeline validation passed")
    return True

# ---------------------------------------------------------------------------
# 4. Launch training
# ---------------------------------------------------------------------------

def launch_training(auto_resume: bool = True) -> subprocess.Popen:
    """Launch training as a subprocess. Returns Popen object."""
    LOG.info("Launching pretraining...")

    cmd = [
        str(VENV_PYTHON), str(TRAIN_SCRIPT),
        "--config", str(TRAINING_CONFIG),
    ]
    if auto_resume:
        cmd.append("--auto-resume")

    # Run training in background, inheriting stdout/stderr
    LOG.info("  Command: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def wait_for_training(process: subprocess.Popen, poll_interval: int = 30):
    """Monitor training process, auto-resume on crash."""
    max_restarts = 10
    restart_count = 0

    while restart_count <= max_restarts:
        retcode = process.poll()

        if retcode is None:
            # Still running
            _log_training_status()
            time.sleep(poll_interval)
            continue

        if retcode == 0:
            LOG.info("Training completed successfully (exit code 0)")
            return True

        # Training crashed — auto-resume
        restart_count += 1
        LOG.warning("Training process exited with code %d (restart %d/%d)",
                    retcode, restart_count, max_restarts)

        if restart_count > max_restarts:
            LOG.error("Max restarts exceeded")
            return False

        LOG.info("Auto-resuming training in 10 seconds...")
        time.sleep(10)
        process = launch_training(auto_resume=True)

    return False


def _log_training_status():
    """Check for latest training log and print status."""
    # Find latest experiment
    exp_dirs = sorted(EXPERIMENT_DIR.glob("2026-*"))
    if not exp_dirs:
        return
    latest = exp_dirs[-1]
    log_file = latest / "metrics.csv"
    if not log_file.exists():
        return

    try:
        with open(log_file) as f:
            lines = f.readlines()
        if len(lines) < 2:
            return
        last_line = lines[-1].strip()
        LOG.info("  Latest metrics: %s", last_line)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 5. Post-training evaluation
# ---------------------------------------------------------------------------

def run_evaluation() -> dict:
    """Run the evaluation suite on the best checkpoint."""
    LOG.info("Running evaluation suite...")

    # Find latest experiment
    exp_dirs = sorted(EXPERIMENT_DIR.glob("2026-*"))
    if not exp_dirs:
        LOG.error("No experiment directories found")
        return {}
    exp_dir = exp_dirs[-1]
    LOG.info("  Experiment: %s", exp_dir)

    # Find best checkpoint
    ckpt_dir = exp_dir / "checkpoints"
    best_ckpt = ckpt_dir / "best_model"
    if not best_ckpt.exists():
        # Find latest step checkpoint
        step_ckpts = sorted(ckpt_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
        if step_ckpts:
            best_ckpt = step_ckpts[-1]
        else:
            LOG.error("No checkpoints found")
            return {}

    LOG.info("  Checkpoint: %s", best_ckpt)

    # Run evaluation
    eval_output = exp_dir / "evaluation"
    eval_output.mkdir(exist_ok=True)

    result = subprocess.run(
        [
            str(VENV_PYTHON), str(EVAL_SCRIPT),
            "--checkpoint", str(best_ckpt),
            "--experiment-dir", str(exp_dir),
            "--config", str(TRAINING_CONFIG),
        ],
        capture_output=True, text=True, timeout=600,
        cwd=str(PROJECT_ROOT)
    )

    LOG.info("Evaluation output:\n%s", result.stdout[-2000:] if result.stdout else "(empty)")
    if result.returncode != 0:
        LOG.error("Evaluation failed:\n%s", result.stderr[-2000:])
        # Try fallback — run inline eval
        return _run_inline_eval(best_ckpt, exp_dir)

    # Parse results
    report_path = eval_output / "report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return {}


def _run_inline_eval(ckpt_path: Path, exp_dir: Path) -> dict:
    """Fallback: run evaluation inline if script not available."""
    LOG.info("Running inline evaluation...")

    code = textwrap.dedent(f"""\
import sys, os, json, torch
sys.path.insert(0, "{PROJECT_ROOT}")

import yaml
with open("{TRAINING_CONFIG}") as f:
    config = yaml.safe_load(f)

from training.trainer import Trainer, evaluate
trainer = Trainer(config)

# Load checkpoint
from training.utils.checkpoint import load_checkpoint
load_checkpoint(
    "{ckpt_path}", trainer.model, trainer.optimizer,
    trainer.scheduler, trainer.scaler, trainer.ema, trainer.device
)
print(f"Loaded checkpoint: {ckpt_path}")

# Run validation
eval_steps = config.get("training", {{}}).get("eval_steps", 100)
results = evaluate(
    trainer.model, trainer.val_dataloader, trainer.device,
    max_steps=eval_steps, pad_token_id=trainer.tokenizer.pad_token_id,
)

# Save results
eval_dir = "{exp_dir}" / "evaluation"
eval_dir = eval_dir if isinstance(eval_dir, __import__('pathlib').Path) else __import__('pathlib').Path(eval_dir)
eval_dir.mkdir(exist_ok=True)
with open(eval_dir / "report.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(json.dumps(results, indent=2, default=str))
""")
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", code],
        capture_output=True, text=True, timeout=300,
        cwd=str(PROJECT_ROOT)
    )
    LOG.info("Inline eval output:\n%s", result.stdout[-2000:])
    if result.returncode != 0:
        LOG.error("Inline eval failed:\n%s", result.stderr[-1000:])
    return {}


def generate_reports(eval_results: dict, baseline_path: Optional[str] = None):
    """Generate report.json, report.md, README.md, comparison, and samples."""
    LOG.info("Generating reports...")

    # Find latest experiment
    exp_dirs = sorted(EXPERIMENT_DIR.glob("2026-*"))
    if not exp_dirs:
        return
    exp_dir = exp_dirs[-1]
    eval_dir = exp_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)

    # report.json
    report_path = eval_dir / "report.json"
    if eval_results:
        with open(report_path, "w") as f:
            json.dump(eval_results, f, indent=2, default=str)

    # report.md
    md_lines = [
        f"# Training Report — {exp_dir.name}",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}\n",
        "## Training Results\n",
    ]
    for key in ["val_loss", "val_perplexity", "val_accuracy", "best_val_loss",
                 "final_val_loss", "final_perplexity"]:
        if key in eval_results:
            md_lines.append(f"- **{key}**: {eval_results[key]}")
    md_lines.append(f"\n## Checkpoint\n- Path: `{exp_dir / 'checkpoints'}`\n")

    with open(eval_dir / "report.md", "w") as f:
        f.write("\n".join(md_lines))

    # README.md
    with open(exp_dir / "README.md", "w") as f:
        f.write(f"# {exp_dir.name}\n\n")
        f.write("## Config\n```yaml\n")
        f.write(json.dumps(eval_results, indent=2, default=str))
        f.write("\n```\n")

    # Update leaderboard
    _update_leaderboard(exp_dir, eval_results)

    LOG.info("Reports generated in %s", eval_dir)


def _update_leaderboard(exp_dir: Path, eval_results: dict):
    """Add entry to leaderboard CSV."""
    lb_path = PROJECT_ROOT / "leaderboard.csv"
    import csv

    row = {
        "experiment": exp_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "val_loss": eval_results.get("val_loss", eval_results.get("best_val_loss", "")),
        "val_perplexity": eval_results.get("val_perplexity", eval_results.get("best_perplexity", "")),
        "val_accuracy": eval_results.get("val_accuracy", ""),
        "benchmark_hellaswag": eval_results.get("hellaswag_accuracy", ""),
        "benchmark_boolq": eval_results.get("boolq_accuracy", ""),
        "benchmark_winogrande": eval_results.get("winogrande_accuracy", ""),
    }

    write_header = not lb_path.exists()
    with open(lb_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    LOG.info("Leaderboard updated: %s", lb_path)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def print_summary(state: dict, eval_results: dict):
    """Print final pipeline summary."""
    elapsed = "unknown"
    if "pipeline_started" in state and "completed" in state:
        t0 = datetime.fromisoformat(state["pipeline_started"])
        t1 = datetime.fromisoformat(state["completed"])
        elapsed = str(t1 - t0)

    print("\n" + "=" * 70)
    print("SMALL MIXTURE TRAINING PIPELINE — COMPLETE")
    print("=" * 70)
    print(f"  Total time:        {elapsed}")
    print(f"  Tokens processed:  {eval_results.get('tokens_processed', 'N/A')}")
    print(f"  Best val loss:     {eval_results.get('best_val_loss', eval_results.get('val_loss', 'N/A'))}")
    print(f"  Perplexity:        {eval_results.get('best_perplexity', eval_results.get('val_perplexity', 'N/A'))}")
    print(f"  HellaSwag:         {eval_results.get('hellaswag_accuracy', 'N/A')}")
    print(f"  BoolQ:             {eval_results.get('boolq_accuracy', 'N/A')}")
    print(f"  Winogrande:        {eval_results.get('winogrande_accuracy', 'N/A')}")
    exp_dirs = sorted(EXPERIMENT_DIR.glob("2026-*"))
    if exp_dirs:
        print(f"  Experiment dir:    {exp_dirs[-1]}")
    print("=" * 70 + "\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SMALL mixture training pipeline")
    parser.add_argument("--skip-build", action="store_true", help="Skip build monitoring")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--build-timeout", type=int, default=7200, help="Build wait timeout (s)")
    args = parser.parse_args()

    setup_logging()
    state = load_state()

    try:
        # Phase 1: Wait for build
        if not args.skip_build and not args.eval_only:
            set_phase(state, "build_monitor")
            if not wait_for_build(timeout=args.build_timeout):
                LOG.error("Build did not complete — aborting")
                sys.exit(1)
            set_phase(state, "build_verify")
            verify_build()

        # Phase 2: Train tokenizer
        if not args.eval_only:
            set_phase(state, "tokenizer")
            tok_stats = train_tokenizer()
            state["tokenizer_stats"] = tok_stats
            save_state(state)
            copy_tokenizer()

            # Phase 3: Validate pipeline
            set_phase(state, "validation")
            validate_pipeline()

            # Phase 4: Train
            set_phase(state, "training")
            process = launch_training(auto_resume=True)
            success = wait_for_training(process)
            if not success:
                LOG.error("Training did not complete successfully")
                # Continue to eval anyway — may have partial results

        # Phase 5: Evaluate
        set_phase(state, "evaluation")
        eval_results = run_evaluation()
        state["eval_results"] = eval_results
        save_state(state)

        # Phase 6: Reports
        set_phase(state, "reports")
        generate_reports(eval_results)
        state["completed"] = datetime.now(timezone.utc).isoformat()
        save_state(state)

        # Final summary
        print_summary(state, eval_results)

    except KeyboardInterrupt:
        LOG.warning("Pipeline interrupted by user")
        save_state(state)
        sys.exit(130)
    except Exception as e:
        LOG.error("Pipeline failed: %s", e, exc_info=True)
        state["error"] = str(e)
        save_state(state)
        sys.exit(1)


if __name__ == "__main__":
    main()
