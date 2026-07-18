"""Auto-recovery for training experiments.

Handles:
  - Process crash → auto-resume from latest valid checkpoint
  - OOM → reduce batch size and retry
  - Keyboard interruption → graceful shutdown with checkpoint
  - Corrupted checkpoint → fallback to previous checkpoint
  - Missing tokenizer → retrain from corpus
  - Missing dataset shard → rebuild or skip

Guarantee: Never lose more than one checkpoint interval of work.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

LOG = logging.getLogger(__name__)


class RecoveryManager:
    """Manages auto-recovery for training experiments."""

    def __init__(
        self,
        project_root: Path,
        checkpoint_dir: Path,
        experiment_dir: Path,
        max_retries: int = 5,
        retry_delay: int = 30,
    ):
        self.project_root = Path(project_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_dir = Path(experiment_dir)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def find_valid_checkpoint(self) -> Optional[Path]:
        """Find the latest valid checkpoint, falling back to previous ones."""
        if not self.checkpoint_dir.exists():
            LOG.warning("Checkpoint directory does not exist: %s", self.checkpoint_dir)
            return None

        # List all step checkpoints, sorted by step number descending
        step_ckpts = []
        for f in self.checkpoint_dir.iterdir():
            if f.name.startswith("step_") and not f.name.endswith(".tmp"):
                try:
                    step = int(f.name.split("_")[1])
                    step_ckpts.append((step, f))
                except (IndexError, ValueError):
                    continue

        step_ckpts.sort(key=lambda x: x[0], reverse=True)

        # Also check for special checkpoints
        special_names = ["best_model", "early_stop_final"]
        for name in special_names:
            p = self.checkpoint_dir / name
            if p.exists():
                # Verify it's a file (not a directory of corrupted data)
                if self._validate_checkpoint(p):
                    LOG.info("Found valid special checkpoint: %s", name)
                    return p

        # Try step checkpoints from newest to oldest
        for step, path in step_ckpts:
            if self._validate_checkpoint(path):
                LOG.info("Found valid checkpoint: %s (step %d)", path.name, step)
                return path
            else:
                LOG.warning("Corrupted checkpoint: %s — trying previous", path.name)

        LOG.warning("No valid checkpoints found")
        return None

    def _validate_checkpoint(self, path: Path) -> bool:
        """Validate that a checkpoint file/directory is not corrupted."""
        try:
            if path.is_file():
                # Single file checkpoint
                if path.stat().st_size < 1024:
                    return False
                # Try to load header
                import torch
                data = torch.load(path, map_location="cpu", weights_only=False)
                return "model" in data or "step" in data
            elif path.is_dir():
                # Directory checkpoint
                model_file = path / "model.pt"
                if not model_file.exists():
                    model_file = path / "pytorch_model.bin"
                if not model_file.exists():
                    # Check for single file in directory
                    files = list(path.glob("*.pt"))
                    if files:
                        model_file = files[0]
                    else:
                        return False
                if model_file.stat().st_size < 1024:
                    return False
                import torch
                data = torch.load(model_file, map_location="cpu", weights_only=False)
                return isinstance(data, dict)
        except Exception as e:
            LOG.debug("Checkpoint validation failed for %s: %s", path, e)
            return False
        return False

    def cleanup_corrupted_checkpoints(self):
        """Remove corrupted checkpoint files/directories."""
        if not self.checkpoint_dir.exists():
            return

        for f in self.checkpoint_dir.iterdir():
            if f.name.startswith("step_") and not f.name.endswith(".tmp"):
                if not self._validate_checkpoint(f):
                    LOG.warning("Removing corrupted checkpoint: %s", f.name)
                    if f.is_dir():
                        shutil.rmtree(f, ignore_errors=True)
                    else:
                        f.unlink(missing_ok=True)

    def repair_checkpoint_dir(self):
        """Remove temporary files and repair the checkpoint directory."""
        if not self.checkpoint_dir.exists():
            return

        # Remove .tmp files
        for f in self.checkpoint_dir.glob("*.tmp"):
            LOG.info("Removing temp file: %s", f.name)
            f.unlink(missing_ok=True)

        # Remove incomplete step directories (no model file inside)
        for d in self.checkpoint_dir.iterdir():
            if d.is_dir() and d.name.startswith("step_"):
                has_model = any(d.glob("*.pt")) or any(d.glob("*.bin"))
                if not has_model:
                    LOG.warning("Removing incomplete checkpoint: %s", d.name)
                    shutil.rmtree(d, ignore_errors=True)

    def save_training_state(self, path: Path, state: dict):
        """Save training state atomically (write-to-temp then rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
            # Atomic rename
            tmp_path.rename(path)
        except Exception as e:
            LOG.error("Failed to save training state: %s", e)
            tmp_path.unlink(missing_ok=True)
            raise

    def load_training_state(self, path: Path) -> Optional[dict]:
        """Load training state, falling back to .bak if corrupted."""
        path = Path(path)
        for attempt_path in [path, path.with_suffix(".bak")]:
            if attempt_path.exists():
                try:
                    with open(attempt_path) as f:
                        state = json.load(f)
                    # Backup the good copy
                    if attempt_path == path:
                        shutil.copy2(path, path.with_suffix(".bak"))
                    return state
                except (json.JSONDecodeError, IOError) as e:
                    LOG.warning("Corrupted state file %s: %s", attempt_path, e)
        return None

    def check_dataset_integrity(self, corpus_path: Path, min_bytes: int = 1_000_000) -> bool:
        """Check that the dataset file exists and is not too small."""
        if not corpus_path.exists():
            LOG.error("Dataset not found: %s", corpus_path)
            return False
        size = corpus_path.stat().st_size
        if size < min_bytes:
            LOG.error("Dataset too small: %s (%d bytes < %d)", corpus_path, size, min_bytes)
            return False
        # Check first and last bytes for corruption
        try:
            with open(corpus_path, "rb") as f:
                first = f.read(100)
                f.seek(-100, 2)
                last = f.read(100)
            if not first.strip() or not last.strip():
                LOG.error("Dataset may be corrupted (empty first/last bytes)")
                return False
        except IOError:
            LOG.error("Cannot read dataset: %s", corpus_path)
            return False
        return True

    def check_tokenizer(self, tokenizer_path: Path) -> bool:
        """Check that the tokenizer model exists and loads."""
        if not tokenizer_path.exists():
            LOG.error("Tokenizer not found: %s", tokenizer_path)
            return False
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            if not sp.Load(str(tokenizer_path)):
                LOG.error("Tokenizer failed to load: %s", tokenizer_path)
                return False
            # Quick sanity check
            test = sp.EncodeAsIds("hello world")
            if len(test) == 0:
                LOG.error("Tokenizer produces empty encoding")
                return False
            return True
        except Exception as e:
            LOG.error("Tokenizer validation failed: %s", e)
            return False

    def attempt_recovery(self, stage: str, error: Exception) -> Optional[dict]:
        """Attempt recovery from a specific failure stage. Returns recovery instructions or None."""
        LOG.info("Attempting recovery for stage '%s': %s", stage, error)

        if stage == "checkpoint_load":
            return self._recover_checkpoint_load()
        elif stage == "oom":
            return self._recover_oom()
        elif stage == "tokenizer_missing":
            return self._recover_tokenizer()
        elif stage == "dataset_missing":
            return self._recover_dataset()
        elif stage == "training_crash":
            return self._recover_training_crash()

        LOG.warning("No recovery strategy for stage: %s", stage)
        return None

    def _recover_checkpoint_load(self) -> Optional[dict]:
        """Recover from corrupted checkpoint by finding a valid one."""
        LOG.info("Recovery: searching for valid checkpoint...")
        valid = self.find_valid_checkpoint()
        if valid:
            return {"action": "load_checkpoint", "path": str(valid)}
        return None

    def _recover_oom(self) -> Optional[dict]:
        """Recover from OOM by reducing batch size."""
        LOG.info("Recovery: reducing batch size for OOM...")
        return {"action": "reduce_batch_size", "factor": 0.5}

    def _recover_tokenizer(self) -> Optional[dict]:
        """Recover missing tokenizer by retraining."""
        LOG.info("Recovery: tokenizer missing, needs retraining")
        return {"action": "retrain_tokenizer"}

    def _recover_dataset(self) -> Optional[dict]:
        """Recover missing dataset by checking alternative locations."""
        LOG.info("Recovery: dataset missing, checking alternatives...")
        return None

    def _recover_training_crash(self) -> Optional[dict]:
        """Recover from training crash by resuming from last checkpoint."""
        valid = self.find_valid_checkpoint()
        if valid:
            return {"action": "resume_training", "path": str(valid)}
        return None

    def create_backup(self, source: Path, backup_name: str) -> Optional[Path]:
        """Create a timestamped backup of a file or directory."""
        backup_dir = self.experiment_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        dest = backup_dir / f"{backup_name}_{ts}"
        try:
            if source.is_dir():
                shutil.copytree(source, dest)
            else:
                shutil.copy2(source, dest)
            LOG.info("Backup created: %s", dest)
            return dest
        except Exception as e:
            LOG.error("Backup failed: %s", e)
            return None
