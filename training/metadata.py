"""Experiment metadata collection.

Automatically records:
  - Git commit and diff
  - Python, PyTorch, macOS versions
  - Apple Silicon chip info
  - RAM and MPS availability
  - Training configuration
  - Dataset and tokenizer metadata
  - Random seeds and timestamps
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

LOG = logging.getLogger(__name__)


def collect_experiment_metadata(
    project_root: Path,
    config: dict,
    tokenizer_info: Optional[dict] = None,
    dataset_info: Optional[dict] = None,
) -> dict:
    """Collect comprehensive experiment metadata."""
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "project_root": str(project_root),
        "git": _collect_git_info(project_root),
        "system": _collect_system_info(),
        "python": _collect_python_info(),
        "ml_framework": _collect_ml_framework_info(),
        "config": _sanitize_config(config),
        "seeds": _collect_seeds(config),
    }

    if tokenizer_info:
        meta["tokenizer"] = tokenizer_info
    if dataset_info:
        meta["dataset"] = dataset_info

    return meta


def save_experiment_metadata(output_dir: Path, metadata: dict):
    """Save metadata to experiment directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / "experiment_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    LOG.info("Experiment metadata saved: %s", meta_path)

    # Also save a human-readable summary
    summary_path = output_dir / "experiment_summary.txt"
    with open(summary_path, "w") as f:
        f.write(_format_metadata_summary(metadata))
    LOG.info("Experiment summary saved: %s", summary_path)


def _collect_git_info(project_root: Path) -> dict:
    """Collect git repository information."""
    info: dict[str, Any] = {}

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(project_root),
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(project_root),
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
            cwd=str(project_root),
        )
        if result.returncode == 0:
            dirty = result.stdout.strip()
            info["dirty"] = bool(dirty)
            if dirty:
                info["dirty_files"] = dirty.split("\n")[:20]  # first 20
                # Save full diff
                diff_result = subprocess.run(
                    ["git", "diff", "--stat"],
                    capture_output=True, text=True, timeout=10,
                    cwd=str(project_root),
                )
                if diff_result.returncode == 0:
                    info["diff_stat"] = diff_result.stdout.strip()
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-10"],
            capture_output=True, text=True, timeout=5,
            cwd=str(project_root),
        )
        if result.returncode == 0:
            info["recent_commits"] = result.stdout.strip().split("\n")
    except Exception:
        pass

    return info


def _collect_system_info() -> dict:
    """Collect system information."""
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "node": platform.node(),
    }

    # Apple Silicon specific
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            info["chip"] = result.stdout.strip()
    except Exception:
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.optional.arm64"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0 and result.stdout.strip() == "1":
                info["chip"] = "Apple Silicon (ARM64)"
        except Exception:
            pass

    # RAM
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            ram_bytes = int(result.stdout.strip())
            info["ram_gb"] = round(ram_bytes / 1e9, 1)
    except Exception:
        pass

    # CPU cores
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.ncpu"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            info["cpu_cores"] = int(result.stdout.strip())
    except Exception:
        pass

    return info


def _collect_python_info() -> dict:
    """Collect Python and package information."""
    info: dict[str, Any] = {
        "version": sys.version,
        "implementation": sys.implementation.name,
        "executable": sys.executable,
    }

    # Key package versions
    packages = ["torch", "datasets", "sentencepiece", "transformers", "numpy", "psutil"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{pkg}_version"] = "not installed"

    return info


def _collect_ml_framework_info() -> dict:
    """Collect ML framework information."""
    info: dict[str, Any] = {}

    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = hasattr(torch, "mps") and torch.mps.is_available()

        if info["mps_available"]:
            info["mps_built"] = torch.backends.mps.is_built()
            info["mps_macos_version"] = str(torch.backends.mps.macos_version()) if hasattr(torch.backends.mps, "macos_version") else "unknown"
            # Memory
            try:
                info["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
            except Exception:
                pass
            try:
                if hasattr(torch.mps, "driver_allocated_memory"):
                    info["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
            except Exception:
                pass
    except ImportError:
        pass

    try:
        import torch
        info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    except Exception:
        pass

    return info


def _collect_seeds(config: dict) -> dict:
    """Extract random seed configurations."""
    training = config.get("training", {})
    return {
        "global_seed": training.get("seed", "not set"),
    }


def _sanitize_config(config: dict) -> dict:
    """Remove sensitive or overly large values from config for metadata."""
    import copy
    sanitized = copy.deepcopy(config)

    # Remove any API keys or tokens
    def _remove_keys(d):
        if isinstance(d, dict):
            for key in list(d.keys()):
                if any(s in key.lower() for s in ["key", "token", "secret", "password"]):
                    d[key] = "***REDACTED***"
                else:
                    _remove_keys(d[key])
        elif isinstance(d, list):
            for item in d:
                _remove_keys(item)

    _remove_keys(sanitized)
    return sanitized


def _format_metadata_summary(metadata: dict) -> str:
    """Format a human-readable metadata summary."""
    lines = ["=" * 60]
    lines.append("EXPERIMENT METADATA SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    # System
    sys_info = metadata.get("system", {})
    lines.append(f"System:    {sys_info.get('os', '?')} {sys_info.get('os_release', '?')}")
    lines.append(f"Chip:      {sys_info.get('chip', 'unknown')}")
    lines.append(f"RAM:       {sys_info.get('ram_gb', '?')} GB")
    lines.append(f"CPU cores: {sys_info.get('cpu_cores', '?')}")
    lines.append("")

    # ML framework
    ml = metadata.get("ml_framework", {})
    lines.append(f"PyTorch:   {ml.get('pytorch_version', '?')}")
    lines.append(f"MPS:       {ml.get('mps_available', '?')}")
    lines.append(f"CUDA:      {ml.get('cuda_available', '?')}")
    lines.append("")

    # Python
    py = metadata.get("python", {})
    lines.append(f"Python:    {py.get('version', '?').split()[0]}")
    lines.append(f"Torch:     {py.get('torch_version', '?')}")
    lines.append(f"Datasets:  {py.get('datasets_version', '?')}")
    lines.append(f"SP:        {py.get('sentencepiece_version', '?')}")
    lines.append("")

    # Git
    git = metadata.get("git", {})
    lines.append(f"Commit:    {git.get('commit', 'N/A')[:12]}")
    lines.append(f"Branch:    {git.get('branch', 'N/A')}")
    lines.append(f"Dirty:     {git.get('dirty', 'N/A')}")
    lines.append("")

    # Seeds
    seeds = metadata.get("seeds", {})
    lines.append(f"Seed:      {seeds.get('global_seed', 'N/A')}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
