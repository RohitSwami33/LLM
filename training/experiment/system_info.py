"""System information collection for reproducibility.

Captures Python version, PyTorch version, CUDA/MPS availability,
OS info, git state, and hardware details.
"""

import os
import sys
import platform
import subprocess
from typing import Dict, Any


class SystemInfo:
    """Collects and stores system environment information."""

    @staticmethod
    def collect_all() -> Dict[str, Any]:
        """Collect complete system information."""
        info = {
            "python": SystemInfo.python_info(),
            "pytorch": SystemInfo.pytorch_info(),
            "cuda": SystemInfo.cuda_info(),
            "mps": SystemInfo.mps_info(),
            "os": SystemInfo.os_info(),
            "hardware": SystemInfo.hardware_info(),
            "git": SystemInfo.git_info(),
        }
        return info

    @staticmethod
    def python_info() -> Dict[str, str]:
        return {
            "version": sys.version,
            "version_short": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "implementation": sys.implementation.name,
            "compiler": sys.version_info[4],
        }

    @staticmethod
    def pytorch_info() -> Dict[str, Any]:
        import torch
        return {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        }

    @staticmethod
    def cuda_info() -> Dict[str, Any]:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "version": torch.version.cuda or "N/A",
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A",
            "compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)) if torch.cuda.device_count() > 0 else "N/A",
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A",
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2) if torch.cuda.device_count() > 0 else 0,
        }

    @staticmethod
    def mps_info() -> Dict[str, Any]:
        import torch
        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return {
            "available": available,
            "built": hasattr(torch.backends, "mps") and torch.backends.mps.is_built(),
        }

    @staticmethod
    def os_info() -> Dict[str, str]:
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "hostname": platform.node(),
        }

    @staticmethod
    def hardware_info() -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        try:
            import os
            info["cpu_count"] = os.cpu_count()
        except Exception:
            info["cpu_count"] = "unknown"

        # Try to get total RAM
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
                info["total_ram_gb"] = round(int(result.stdout.strip()) / 1e9, 2)
            elif platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if "MemTotal" in line:
                            kb = int(line.split()[1])
                            info["total_ram_gb"] = round(kb / 1e6, 2)
                            break
            else:
                info["total_ram_gb"] = "unknown"
        except Exception:
            info["total_ram_gb"] = "unknown"

        return info

    @staticmethod
    def git_info() -> Dict[str, Any]:
        """Get git commit hash and repo info if available."""
        info: Dict[str, Any] = {
            "available": False,
            "commit_hash": None,
            "commit_message": None,
            "branch": None,
            "repo_url": None,
            "dirty": None,
        }

        # Walk up to find .git
        start = os.path.dirname(os.path.abspath(__file__))
        for _ in range(10):
            if os.path.exists(os.path.join(start, ".git")):
                break
            parent = os.path.dirname(start)
            if parent == start:
                return info
            start = parent
        else:
            return info

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5, cwd=start
            )
            if result.returncode == 0:
                info["available"] = True
                info["commit_hash"] = result.stdout.strip()

            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%s"],
                capture_output=True, text=True, timeout=5, cwd=start
            )
            if result.returncode == 0:
                info["commit_message"] = result.stdout.strip()

            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5, cwd=start
            )
            if result.returncode == 0:
                info["branch"] = result.stdout.strip()

            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5, cwd=start
            )
            if result.returncode == 0:
                info["repo_url"] = result.stdout.strip()

            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5, cwd=start
            )
            if result.returncode == 0:
                info["dirty"] = len(result.stdout.strip()) > 0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return info

    @staticmethod
    def save_git_commit_file(path: str) -> None:
        """Save git commit hash to a text file."""
        git = SystemInfo.git_info()
        with open(path, "w") as f:
            if git["available"]:
                f.write(f"commit: {git['commit_hash']}\n")
                f.write(f"branch: {git['branch']}\n")
                f.write(f"message: {git['commit_message']}\n")
                f.write(f"repo: {git['repo_url']}\n")
                f.write(f"dirty: {git['dirty']}\n")
            else:
                f.write("git info not available\n")

    @staticmethod
    def format_summary(info: Dict[str, Any]) -> str:
        """Format system info as readable text."""
        lines = [
            "=" * 60,
            "SYSTEM INFORMATION",
            "=" * 60,
            f"  OS:            {info['os']['platform']}",
            f"  Python:        {info['python']['version_short']}",
            f"  PyTorch:       {info['pytorch']['version']}",
        ]
        if info["cuda"]["available"]:
            lines.extend([
                f"  CUDA:          {info['cuda']['version']}",
                f"  GPU:           {info['cuda']['device_name']}",
                f"  GPU Memory:    {info['cuda']['total_memory_gb']} GB",
                f"  Compute:       {info['cuda']['compute_capability']}",
            ])
        if info["mps"]["available"]:
            lines.append("  MPS:           available")
        lines.extend([
            f"  CPU Count:     {info['hardware']['cpu_count']}",
            f"  Total RAM:     {info['hardware']['total_ram_gb']} GB",
        ])
        if info["git"]["available"]:
            lines.extend([
                f"  Git Commit:    {info['git']['commit_hash'][:12]}",
                f"  Git Branch:    {info['git']['branch']}",
                f"  Git Dirty:     {info['git']['dirty']}",
            ])
        lines.append("=" * 60)
        return "\n".join(lines)
