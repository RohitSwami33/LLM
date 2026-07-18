"""Health monitoring for long-running training experiments.

Checks every N seconds:
  - Process alive (build, training)
  - MPS/GPU utilization and memory
  - CPU utilization
  - System memory usage
  - Disk usage and free space
  - Log file growth (stall detection)
  - Checkpoint freshness (stale checkpoint detection)

Emits configurable warnings when thresholds are exceeded.
Runs as a daemon thread — never blocks the main training loop.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

LOG = logging.getLogger(__name__)


@dataclass
class HealthThresholds:
    max_cpu_percent: float = 95.0
    max_memory_percent: float = 90.0
    max_mps_memory_gb: float = 14.0
    min_disk_free_gb: float = 5.0
    max_disk_percent: float = 95.0
    log_stall_seconds: int = 300        # no log growth for 5 min = stall
    checkpoint_stale_seconds: int = 1800  # no checkpoint for 30 min = stale
    max_process_memory_gb: float = 14.0


@dataclass
class HealthStatus:
    timestamp: float = 0.0
    build_alive: bool = False
    training_alive: bool = False
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_percent: float = 0.0
    mps_allocated_gb: float = 0.0
    mps_reserved_gb: float = 0.0
    log_size_bytes: int = 0
    log_growth_bytes: int = 0
    checkpoint_age_seconds: float = 0.0
    last_checkpoint: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "build_alive": self.build_alive,
            "training_alive": self.training_alive,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": round(self.memory_used_gb, 2),
            "memory_total_gb": round(self.memory_total_gb, 2),
            "disk_free_gb": round(self.disk_free_gb, 2),
            "disk_percent": self.disk_percent,
            "mps_allocated_gb": round(self.mps_allocated_gb, 3),
            "mps_reserved_gb": round(self.mps_reserved_gb, 3),
            "log_growth_bytes": self.log_growth_bytes,
            "checkpoint_age_seconds": round(self.checkpoint_age_seconds, 0),
            "last_checkpoint": self.last_checkpoint,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class HealthMonitor:
    """Daemon health monitor that checks system and process health."""

    def __init__(
        self,
        project_root: Path,
        thresholds: Optional[HealthThresholds] = None,
        check_interval: int = 60,
        build_pid: Optional[int] = None,
        training_pid: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
        log_file: Optional[Path] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.project_root = Path(project_root)
        self.thresholds = thresholds or HealthThresholds()
        self.check_interval = check_interval
        self._build_pid = build_pid
        self._training_pid = training_pid
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._log_file = Path(log_file) if log_file else None
        self._on_warning = on_warning or (lambda msg: LOG.warning("HEALTH: %s", msg))
        self._on_error = on_error or (lambda msg: LOG.error("HEALTH: %s", msg))

        self._status = HealthStatus()
        self._prev_log_size = 0
        self._prev_log_check_time = time.time()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._history: list[dict] = []
        self._lock = threading.Lock()

    @property
    def status(self) -> HealthStatus:
        with self._lock:
            return self._status

    @property
    def history(self) -> list[dict]:
        with self._lock:
            return list(self._history)

    def set_build_pid(self, pid: Optional[int]):
        self._build_pid = pid

    def set_training_pid(self, pid: Optional[int]):
        self._training_pid = pid

    def start(self):
        """Start the health monitor as a daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="health-monitor")
        self._thread.start()
        LOG.info("Health monitor started (interval=%ds)", self.check_interval)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        LOG.info("Health monitor stopped")

    def check_now(self) -> HealthStatus:
        """Run a single health check immediately (blocking)."""
        return self._run_check()

    def _monitor_loop(self):
        while self._running:
            try:
                self._run_check()
            except Exception as e:
                LOG.error("Health check failed: %s", e)
            time.sleep(self.check_interval)

    def _run_check(self) -> HealthStatus:
        s = HealthStatus(timestamp=time.time())
        ts = self.thresholds

        # Process checks
        s.build_alive = self._is_pid_alive(self._build_pid)
        s.training_alive = self._is_pid_alive(self._training_pid)

        # CPU & memory
        try:
            import psutil
            proc = psutil.Process()
            s.cpu_percent = proc.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            s.memory_percent = mem.percent
            s.memory_used_gb = mem.used / 1e9
            s.memory_total_gb = mem.total / 1e9
            disk = shutil.disk_usage(str(self.project_root))
            s.disk_free_gb = disk.free / 1e9
            s.disk_percent = (disk.used / disk.total) * 100
        except ImportError:
            # Fallback without psutil
            s.cpu_percent = self._get_cpu_percent_fallback()
            s.memory_percent, s.memory_used_gb, s.memory_total_gb = self._get_memory_fallback()
            disk = shutil.disk_usage(str(self.project_root))
            s.disk_free_gb = disk.free / 1e9
            s.disk_percent = (disk.used / disk.total) * 100

        # MPS memory
        try:
            import torch
            if hasattr(torch, "mps") and torch.mps.is_available():
                s.mps_allocated_gb = torch.mps.current_allocated_memory() / 1e9
                if hasattr(torch.mps, "driver_allocated_memory"):
                    s.mps_reserved_gb = torch.mps.driver_allocated_memory() / 1e9
        except Exception:
            pass

        # Log growth (stall detection)
        if self._log_file and self._log_file.exists():
            try:
                current_size = self._log_file.stat().st_size
                now = time.time()
                elapsed = now - self._prev_log_check_time
                if elapsed > 0:
                    s.log_growth_bytes = int((current_size - self._prev_log_size) / max(elapsed, 1))
                self._prev_log_size = current_size
                self._prev_log_check_time = now
                s.log_size_bytes = current_size
            except OSError:
                pass

        # Checkpoint freshness
        if self._checkpoint_dir and self._checkpoint_dir.exists():
            try:
                step_ckpts = sorted(
                    [f for f in self._checkpoint_dir.iterdir() if f.name.startswith("step_")],
                    key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0,
                )
                if step_ckpts:
                    latest = step_ckpts[-1]
                    s.last_checkpoint = latest.name
                    age = time.time() - latest.stat().st_mtime
                    s.checkpoint_age_seconds = age
            except (OSError, ValueError):
                pass

        # Threshold checks
        if s.cpu_percent > ts.max_cpu_percent:
            s.warnings.append(f"CPU usage {s.cpu_percent:.1f}% > {ts.max_cpu_percent}%")
        if s.memory_percent > ts.max_memory_percent:
            s.warnings.append(f"Memory usage {s.memory_percent:.1f}% > {ts.max_memory_percent}%")
        if s.mps_allocated_gb > ts.max_mps_memory_gb:
            s.warnings.append(f"MPS memory {s.mps_allocated_gb:.2f}GB > {ts.max_mps_memory_gb}GB")
        if s.disk_free_gb < ts.min_disk_free_gb:
            s.warnings.append(f"Free disk {s.disk_free_gb:.1f}GB < {ts.min_disk_free_gb}GB")
        if s.disk_percent > ts.max_disk_percent:
            s.warnings.append(f"Disk usage {s.disk_percent:.1f}% > {ts.max_disk_percent}%")
        if s.log_growth_bytes == 0 and self._log_file and self._log_file.exists():
            if time.time() - self._prev_log_check_time > ts.log_stall_seconds:
                s.warnings.append(f"No log growth for {ts.log_stall_seconds}s (possible stall)")
        if (s.checkpoint_age_seconds > ts.checkpoint_stale_seconds and
                s.training_alive and s.checkpoint_age_seconds > 0):
            s.warnings.append(
                f"Last checkpoint {s.checkpoint_age_seconds:.0f}s ago > {ts.checkpoint_stale_seconds}s"
            )

        # Critical: training dead but build is done
        if not s.training_alive and s.build_alive:
            s.warnings.append("Build alive but training not running")
        if not s.build_alive and not s.training_alive and self._build_pid:
            s.errors.append("Both build and training processes dead")

        # Emit warnings
        for w in s.warnings:
            self._on_warning(w)
        for e in s.errors:
            self._on_error(e)

        with self._lock:
            self._status = s
            self._history.append(s.to_dict())
            # Keep last 24 hours of checks (1440 at 60s interval)
            if len(self._history) > 1500:
                self._history = self._history[-1500:]

        return s

    def _is_pid_alive(self, pid: Optional[int]) -> bool:
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _get_cpu_percent_fallback(self) -> float:
        """Fallback CPU measurement using /proc/loadavg or top."""
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True, text=True, timeout=2
            )
            ncpu = int(result.stdout.strip()) if result.returncode == 0 else 1
            load = os.getloadavg()[0]
            return min(100.0, (load / ncpu) * 100)
        except Exception:
            return 0.0

    def _get_memory_fallback(self) -> tuple[float, float, float]:
        """Fallback memory using vm_stat."""
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2
            )
            total = int(result.stdout.strip()) if result.returncode == 0 else 8 * 1e9
            result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2)
            pages_free = 0
            pages_active = 0
            for line in result.stdout.split("\n"):
                if "Pages free" in line:
                    pages_free = int(line.split(":")[1].strip().rstrip(".")) * 4096
                elif "Pages active" in line:
                    pages_active = int(line.split(":")[1].strip().rstrip(".")) * 4096
            used = total - pages_free
            return (used / total) * 100, used / 1e9, total / 1e9
        except Exception:
            return 0.0, 0.0, 8.0


def get_health_summary(status: HealthStatus) -> str:
    """Format a one-line health summary."""
    parts = []
    if status.training_alive:
        parts.append("TRAIN:OK")
    elif status.build_alive:
        parts.append("BUILD:OK")
    else:
        parts.append("DEAD")

    parts.append(f"CPU:{status.cpu_percent:.0f}%")
    parts.append(f"MEM:{status.memory_percent:.0f}%")
    parts.append(f"DISK:{status.disk_free_gb:.1f}GB free")
    if status.mps_allocated_gb > 0:
        parts.append(f"MPS:{status.mps_allocated_gb:.1f}GB")

    if status.warnings:
        parts.append(f"WARN:{len(status.warnings)}")
    if status.errors:
        parts.append(f"ERR:{len(status.errors)}")

    return " | ".join(parts)
