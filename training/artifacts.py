"""Experiment artifact management.

Manages:
  - Checkpoint copying to experiment directory (not symlinks)
  - Log file organization
  - Metadata file creation
  - Reproducibility package generation
  - Artifact integrity verification
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

LOG = logging.getLogger(__name__)


class ArtifactManager:
    """Manages experiment artifacts and reproducibility."""

    def __init__(
        self,
        project_root: Path,
        experiment_dir: Path,
        checkpoint_dir: Path,
    ):
        self.project_root = Path(project_root)
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.artifacts_dir = self.experiment_dir / "artifacts"
        self.checkpoints_copy = self.experiment_dir / "checkpoints"
        self.logs_dir = self.experiment_dir / "logs"

    def setup_experiment_dir(self):
        """Create experiment directory structure."""
        for d in [self.artifacts_dir, self.checkpoints_copy, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
        LOG.info("Experiment directory created: %s", self.experiment_dir)

    def copy_checkpoint(self, checkpoint_path: Path, name: Optional[str] = None) -> Optional[Path]:
        """Copy a checkpoint to the experiment directory."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            LOG.warning("Checkpoint does not exist: %s", checkpoint_path)
            return None

        if name is None:
            name = checkpoint_path.name

        dest = self.checkpoints_copy / name
        try:
            if checkpoint_path.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(checkpoint_path, dest)
            else:
                shutil.copy2(checkpoint_path, dest)
            LOG.info("Checkpoint copied: %s -> %s", checkpoint_path, dest)
            return dest
        except Exception as e:
            LOG.error("Failed to copy checkpoint: %s", e)
            return None

    def copy_best_checkpoint(self, best_checkpoint_path: Optional[Path] = None) -> Optional[Path]:
        """Copy the best checkpoint to experiment directory."""
        if best_checkpoint_path is None:
            best_checkpoint_path = self.checkpoint_dir / "best_model"
        return self.copy_checkpoint(best_checkpoint_path, "best_model")

    def copy_logs(self, log_files: list[Path]):
        """Copy log files to experiment directory."""
        for log_file in log_files:
            log_file = Path(log_file)
            if log_file.exists():
                dest = self.logs_dir / log_file.name
                try:
                    shutil.copy2(log_file, dest)
                    LOG.info("Log copied: %s", log_file.name)
                except Exception as e:
                    LOG.warning("Failed to copy log %s: %s", log_file.name, e)

    def save_artifact(self, name: str, content: str, subdir: Optional[str] = None):
        """Save a text artifact."""
        if subdir:
            dest = self.artifacts_dir / subdir
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest = self.artifacts_dir

        path = dest / name
        with open(path, "w") as f:
            f.write(content)
        LOG.info("Artifact saved: %s", path)

    def save_json_artifact(self, name: str, data: dict, subdir: Optional[str] = None):
        """Save a JSON artifact."""
        content = json.dumps(data, indent=2, default=str)
        self.save_artifact(name, content, subdir)

    def verify_artifacts(self) -> dict:
        """Verify integrity of all experiment artifacts."""
        results = {
            "checkpoints": [],
            "logs": [],
            "artifacts": [],
            "all_valid": True,
        }

        # Check checkpoints
        for ckpt_dir in self.checkpoints_copy.iterdir():
            if ckpt_dir.is_dir():
                has_model = any(ckpt_dir.glob("*.pt")) or any(ckpt_dir.glob("*.bin"))
                valid = has_model
                results["checkpoints"].append({
                    "name": ckpt_dir.name,
                    "valid": valid,
                    "size_mb": sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file()) / 1e6,
                })
                if not valid:
                    results["all_valid"] = False

        # Check logs
        for log_file in self.logs_dir.iterdir():
            if log_file.is_file():
                valid = log_file.stat().st_size > 0
                results["logs"].append({
                    "name": log_file.name,
                    "valid": valid,
                    "size_kb": log_file.stat().st_size / 1024,
                })
                if not valid:
                    results["all_valid"] = False

        # Check artifacts
        for artifact_file in self.artifacts_dir.rglob("*"):
            if artifact_file.is_file():
                valid = artifact_file.stat().st_size > 0
                results["artifacts"].append({
                    "name": str(artifact_file.relative_to(self.artifacts_dir)),
                    "valid": valid,
                    "size_kb": artifact_file.stat().st_size / 1024,
                })
                if not valid:
                    results["all_valid"] = False

        return results

    def generate_reproducibility_package(self) -> Path:
        """Generate a reproducibility package with everything needed to reproduce the experiment."""
        repro_dir = self.experiment_dir / "reproducibility"
        repro_dir.mkdir(parents=True, exist_ok=True)

        # Copy metadata
        meta_src = self.experiment_dir / "experiment_metadata.json"
        if meta_src.exists():
            shutil.copy2(meta_src, repro_dir / "metadata.json")

        # Copy summary
        summary_src = self.experiment_dir / "experiment_summary.txt"
        if summary_src.exists():
            shutil.copy2(summary_src, repro_dir / "summary.txt")

        # Copy config
        config_src = self.experiment_dir / "config.yaml"
        if config_src.exists():
            shutil.copy2(config_src, repro_dir / "config.yaml")

        # Copy evaluation results
        eval_src = self.experiment_dir / "eval_results.json"
        if eval_src.exists():
            shutil.copy2(eval_src, repro_dir / "eval_results.json")

        # Generate reproducibility instructions
        instructions = self._generate_repro_instructions()
        with open(repro_dir / "README.md", "w") as f:
            f.write(instructions)

        # Copy requirements.txt if exists
        req_src = self.project_root / "requirements.txt"
        if req_src.exists():
            shutil.copy2(req_src, repro_dir / "requirements.txt")

        LOG.info("Reproducibility package generated: %s", repro_dir)
        return repro_dir

    def _generate_repro_instructions(self) -> str:
        """Generate reproducibility instructions."""
        return """# Experiment Reproducibility Package

## How to Reproduce

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure MPS is available (Apple Silicon Mac):
```python
import torch
assert torch.mps.is_available()
```

3. Run the experiment:
```bash
python training/trainer.py \\
    --config training/configs/pretrain_small.yaml \\
    --auto-resume
```

4. Evaluate:
```bash
python scripts/evaluate.py \\
    --checkpoint <path_to_checkpoint> \\
    --tokenizer training/tokenizer.model \\
    --output <output_dir>
```

## Key Settings
- Model: 68.7M params, depth=12, 4 heads, 64d head
- Optimizer: Muon(lr=0.02) + AdamW(lr=1e-4) + LR Warmup(375 steps)
- Regularization: weight_decay=0.05, dropout=0.05, grad_clip=1.0, rỗ regularization
- Context: 512 tokens (full causal mask, no sparsity)
- Training: 1B tokens (1 epoch)

## System Requirements
- macOS with Apple Silicon (MPS)
- PyTorch 2.x with MPS support
- 16 GB RAM minimum
"""

    def get_artifact_summary(self) -> dict:
        """Get a summary of all artifacts."""
        summary = {
            "checkpoints": [],
            "logs": [],
            "artifacts": [],
            "total_size_mb": 0,
        }

        # Checkpoints
        if self.checkpoints_copy.exists():
            for ckpt in self.checkpoints_copy.iterdir():
                if ckpt.is_dir():
                    size = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file()) / 1e6
                    summary["checkpoints"].append({"name": ckpt.name, "size_mb": round(size, 2)})
                    summary["total_size_mb"] += size

        # Logs
        if self.logs_dir.exists():
            for log in self.logs_dir.iterdir():
                if log.is_file():
                    size = log.stat().st_size / 1024
                    summary["logs"].append({"name": log.name, "size_kb": round(size, 1)})
                    summary["total_size_mb"] += size / 1024

        # Artifacts
        if self.artifacts_dir.exists():
            for artifact in self.artifacts_dir.rglob("*"):
                if artifact.is_file():
                    size = artifact.stat().st_size / 1024
                    summary["artifacts"].append({
                        "name": str(artifact.relative_to(self.artifacts_dir)),
                        "size_kb": round(size, 1),
                    })
                    summary["total_size_mb"] += size / 1024

        summary["total_size_mb"] = round(summary["total_size_mb"], 2)
        return summary
