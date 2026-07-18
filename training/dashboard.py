"""Live terminal dashboard for training.

Shows in the terminal:
  - Step/loss curves (ASCII)
  - Token throughput
  - MPS memory usage
  - ETA to completion
  - Experiment health status

Runs as a thread that periodically refreshes.
"""

from __future__ import annotations

import curses
import time
import threading
import logging
from typing import Optional, Callable

LOG = logging.getLogger(__name__)


class TrainingDashboard:
    """Live terminal dashboard for training monitoring."""

    def __init__(self, monitor=None, refresh_interval: float = 2.0):
        self.monitor = monitor
        self.refresh_interval = refresh_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._data: dict = {}
        self._lock = threading.Lock()

    def update(self, **kwargs):
        """Update dashboard data (thread-safe)."""
        with self._lock:
            self._data.update(kwargs)

    def start(self):
        """Start the dashboard in background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the dashboard."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        """Main dashboard loop."""
        try:
            curses.wrapper(self._curses_loop)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            LOG.error("Dashboard error: %s", e)
            self._running = False

    def _curses_loop(self, stdscr):
        """Curses-based dashboard rendering."""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)
        stdscr.timeout(int(self.refresh_interval * 1000))

        while self._running:
            try:
                # Check for quit key
                key = stdscr.getch()
                if key == ord("q"):
                    self._running = False
                    break

                # Get current data
                with self._lock:
                    data = dict(self._data)

                # Clear and redraw
                stdscr.clear()
                height, width = stdscr.getmaxyx()

                self._draw_header(stdscr, data, width)
                self._draw_training_stats(stdscr, data, width)
                self._draw_loss_curve(stdscr, data, height, width)
                self._draw_system_stats(stdscr, data, height, width)
                self._draw_footer(stdscr, width)

                stdscr.refresh()
            except curses.error:
                pass
            except Exception as e:
                LOG.debug("Dashboard render error: %s", e)

    def _draw_header(self, stdscr, data: dict, width: int):
        """Draw dashboard header."""
        phase = data.get("phase", "UNKNOWN")
        step = data.get("step", 0)
        total = data.get("total_steps", 0)

        title = f"  TRAINING DASHBOARD | {phase.upper()}"
        if total > 0:
            pct = (step / total) * 100
            title += f" | {step}/{total} ({pct:.1f}%)"

        title = title[:width - 1]
        stdscr.addstr(0, 0, title, curses.A_REVERSE | curses.A_BOLD)
        if len(title) < width:
            stdscr.addstr(0, len(title), " " * (width - len(title)), curses.A_REVERSE)

    def _draw_training_stats(self, stdscr, data: dict, width: int):
        """Draw training statistics."""
        y = 2
        col_width = width // 3

        # Row 1: Loss and PPL
        loss = data.get("loss", None)
        val_loss = data.get("val_loss", None)
        ppl = data.get("ppl", None)

        loss_str = f"  Loss: {loss:.4f}" if loss is not None else "  Loss: N/A"
        val_str = f"  Val Loss: {val_loss:.4f}" if val_loss is not None else "  Val Loss: N/A"
        ppl_str = f"  PPL: {ppl:.2f}" if ppl is not None else "  PPL: N/A"

        stdscr.addstr(y, 0, loss_str[:col_width - 1])
        stdscr.addstr(y, col_width, val_str[:col_width - 1])
        stdscr.addstr(y, 2 * col_width, ppl_str[:col_width - 1])

        # Row 2: Throughput and timing
        y += 1
        tok_s = data.get("tok_per_sec", None)
        step_time = data.get("step_time", None)
        total_tokens = data.get("total_tokens", None)

        tok_str = f"  Tok/s: {tok_s:.0f}" if tok_s is not None else "  Tok/s: N/A"
        time_str = f"  Step: {step_time:.1f}s" if step_time is not None else "  Step: N/A"
        tok_total_str = f"  Total tok: {total_tokens:,}" if total_tokens is not None else "  Total tok: N/A"

        stdscr.addstr(y, 0, tok_str[:col_width - 1])
        stdscr.addstr(y, col_width, time_str[:col_width - 1])
        stdscr.addstr(y, 2 * col_width, tok_total_str[:col_width - 1])

    def _draw_loss_curve(self, stdscr, data: dict, height: int, width: int):
        """Draw ASCII loss curve."""
        y_start = 5
        x_start = 2
        curve_height = min(8, height - y_start - 8)
        curve_width = min(width - x_start - 4, 60)

        if curve_height < 3 or curve_width < 10:
            return

        losses = data.get("loss_history", [])
        if not losses:
            stdscr.addstr(y_start, x_start, "  Loss: waiting for data...")
            return

        # Draw axis
        stdscr.addstr(y_start, x_start, "+", curses.A_BOLD)
        for x in range(1, curve_width + 1):
            stdscr.addstr(y_start, x_start + x, "-", curses.A_DIM)
        for y in range(1, curve_height + 1):
            stdscr.addstr(y_start + y, x_start, "|", curses.A_DIM)

        # Find loss range
        min_loss = min(losses) if losses else 0
        max_loss = max(losses) if losses else 10
        loss_range = max_loss - min_loss if max_loss != min_loss else 1

        # Sample losses to fit curve
        n_points = min(len(losses), curve_width)
        if len(losses) > n_points:
            indices = [int(i * (len(losses) - 1) / (n_points - 1)) for i in range(n_points)]
            sampled = [losses[i] for i in indices]
        else:
            sampled = losses[-n_points:]
            n_points = len(sampled)

        # Plot points
        for i, loss_val in enumerate(sampled):
            x = x_start + 1 + i
            if x > x_start + curve_width:
                break
            # Normalize y (0 = top, curve_height-1 = bottom)
            norm = (loss_val - min_loss) / loss_range
            y = y_start + 1 + int(norm * (curve_height - 2))
            y = max(y_start + 1, min(y, y_start + curve_height - 1))
            stdscr.addstr(y, x, "*", curses.A_BOLD)

        # Labels
        label_y = y_start + curve_height + 1
        stdscr.addstr(label_y, x_start, f"  Loss curve ({n_points} pts) | min={min_loss:.4f} max={max_loss:.4f}")

    def _draw_system_stats(self, stdscr, data: dict, height: int, width: int):
        """Draw MPS/memory stats."""
        y = height - 6
        if y < 0:
            return

        col_width = width // 3

        # MPS memory
        mps_used = data.get("mps_memory_used", None)
        mps_total = data.get("mps_memory_total", None)
        if mps_used is not None and mps_total is not None and mps_total > 0:
            mps_pct = (mps_used / mps_total) * 100
            mps_str = f"  MPS: {mps_used / 1e6:.1f} MB / {mps_total / 1e6:.1f} MB ({mps_pct:.1f}%)"
        else:
            mps_str = "  MPS: N/A"

        # CPU
        cpu_pct = data.get("cpu_percent", None)
        cpu_str = f"  CPU: {cpu_pct:.1f}%" if cpu_pct is not None else "  CPU: N/A"

        # RAM
        ram_pct = data.get("memory_percent", None)
        ram_used = data.get("memory_used_gb", None)
        ram_str = f"  RAM: {ram_used:.1f} GB ({ram_pct:.1f}%)" if ram_used is not None and ram_pct is not None else "  RAM: N/A"

        stdscr.addstr(y, 0, mps_str[:col_width - 1])
        stdscr.addstr(y, col_width, cpu_str[:col_width - 1])
        stdscr.addstr(y, 2 * col_width, ram_str[:col_width - 1])

        # ETA
        eta = data.get("eta_seconds", None)
        if eta is not None:
            if eta < 60:
                eta_str = f"  ETA: {eta:.0f}s"
            elif eta < 3600:
                eta_str = f"  ETA: {eta/60:.1f}m"
            else:
                eta_str = f"  ETA: {eta/3600:.1f}h"
        else:
            eta_str = "  ETA: N/A"

        status = data.get("health_status", "unknown")
        status_str = f"  Status: {status.upper()}"
        stdscr.addstr(y + 1, 0, eta_str)
        stdscr.addstr(y + 1, col_width, status_str)

    def _draw_footer(self, stdscr, width: int):
        """Draw footer with controls."""
        y = stdscr.getmaxyx()[0] - 1
        if y < 0:
            return
        footer = "  [Q] Quit"
        stdscr.addstr(y, 0, footer[:width - 1], curses.A_DIM)


class SimpleDashboard:
    """Non-curses fallback dashboard that prints to stdout."""

    def __init__(self, monitor=None, refresh_interval: float = 5.0):
        self.monitor = monitor
        self.refresh_interval = refresh_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._data: dict = {}
        self._lock = threading.Lock()
        self._last_print = 0

    def update(self, **kwargs):
        with self._lock:
            self._data.update(kwargs)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while self._running:
            now = time.time()
            if now - self._last_print >= self.refresh_interval:
                with self._lock:
                    data = dict(self._data)
                self._print_status(data)
                self._last_print = now
            time.sleep(0.5)

    def _print_status(self, data: dict):
        """Print a single-line status update."""
        parts = []
        phase = data.get("phase", "")
        step = data.get("step", 0)
        loss = data.get("loss")
        tok_s = data.get("tok_per_sec")
        mps_used = data.get("mps_memory_used")

        if phase:
            parts.append(f"[{phase.upper()}]")
        if step:
            parts.append(f"step={step}")
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if tok_s is not None:
            parts.append(f"{tok_s:.0f} tok/s")
        if mps_used is not None:
            parts.append(f"MPS={mps_used / 1e6:.1f}MB")

        line = " | ".join(parts)
        print(f"\r{line:<80}", end="", flush=True)
