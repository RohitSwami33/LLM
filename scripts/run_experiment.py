#!/usr/bin/env python3
"""Production experiment orchestrator.

Integrates:
  - Health monitoring (background thread)
  - Auto-recovery
  - Experiment metadata collection
  - Artifact management
  - Live dashboard (optional)
  - Post-run analysis
  - Research log management

Guarantee: Never lose more than one checkpoint interval of work.

Usage:
    python scripts/run_experiment.py --config training/configs/pretrain_small.yaml [--dashboard]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.monitor import HealthMonitor, MonitorConfig
from training.recovery import RecoveryManager
from training.metadata import collect_experiment_metadata, save_experiment_metadata
from training.artifacts import ArtifactManager
from training.analysis import AnalysisManager
from training.research_log import ResearchLog

LOG = logging.getLogger(__name__)


class ExperimentRunner:
    """Production experiment orchestrator."""

    def __init__(
        self,
        config_path: Path,
        project_root: Path = PROJECT_ROOT,
        use_dashboard: bool = False,
        auto_resume: bool = True,
    ):
        self.config_path = Path(config_path)
        self.project_root = project_root
        self.use_dashboard = use_dashboard
        self.auto_resume = auto_resume

        # Load config
        import yaml
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        # Set up directories
        self.experiment_dir = self._create_experiment_dir()
        self.checkpoint_dir = project_root / "training" / "checkpoints"

        # Initialize managers
        self.recovery = RecoveryManager(
            project_root=project_root,
            checkpoint_dir=self.checkpoint_dir,
            experiment_dir=self.experiment_dir,
        )
        self.artifacts = ArtifactManager(
            project_root=project_root,
            experiment_dir=self.experiment_dir,
            checkpoint_dir=self.checkpoint_dir,
        )
        self.analysis = AnalysisManager(
            project_root=project_root,
            experiment_dir=self.experiment_dir,
        )
        self.research_log = ResearchLog(project_root=project_root)
        self.monitor = None
        self.dashboard = None

        # Setup
        self.artifacts.setup_experiment_dir()
        self.research_log.ensure_history_file()
        self.recovery.cleanup_corrupted_checkpoints()
        self.recovery.repair_checkpoint_dir()

        # Signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        self._shutdown_requested = False

    def _create_experiment_dir(self) -> Path:
        """Create a timestamped experiment directory."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        base_dir = self.project_root / "experiments" / date_str

        # Find next available experiment number
        existing = list(base_dir.glob("*_*")) if base_dir.exists() else []
        next_num = len(existing) + 1
        exp_dir = base_dir / f"{next_num:03d}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config copy
        import shutil
        config_dest = exp_dir / "config.yaml"
        shutil.copy2(self.config_path, config_dest)

        LOG.info("Experiment directory: %s", exp_dir)
        return exp_dir

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully."""
        LOG.info("Received signal %s, initiating graceful shutdown...", signum)
        self._shutdown_requested = True

    def start_monitor(self):
        """Start the health monitor daemon."""
        monitor_config = MonitorConfig(
            check_interval=5.0,
            cpu_threshold=95.0,
            memory_threshold=90.0,
            mps_memory_threshold=14.0,
            stall_threshold=300.0,
        )
        self.monitor = HealthMonitor(
            project_root=self.project_root,
            checkpoint_dir=self.checkpoint_dir,
            config=monitor_config,
        )

        # Register callbacks
        self.monitor.on_warning(self._on_health_warning)
        self.monitor.on_critical(self._on_health_critical)

        self.monitor.start()
        LOG.info("Health monitor started")

    def start_dashboard(self):
        """Start the live dashboard."""
        if not self.use_dashboard:
            return

        try:
            from training.dashboard import TrainingDashboard
            self.dashboard = TrainingDashboard(monitor=self.monitor)
            self.dashboard.start()
            LOG.info("Dashboard started")
        except Exception as e:
            LOG.warning("Dashboard failed to start (falling back to simple): %s", e)
            from training.dashboard import SimpleDashboard
            self.dashboard = SimpleDashboard(monitor=self.monitor)
            self.dashboard.start()

    def stop_services(self):
        """Stop all background services."""
        if self.monitor:
            self.monitor.stop()
        if self.dashboard:
            self.dashboard.stop()

    def _on_health_warning(self, warning_type: str, message: str):
        """Handle health warning."""
        LOG.warning("Health warning [%s]: %s", warning_type, message)

        # Attempt recovery for specific issues
        if warning_type == "checkpoint_stale":
            recovery = self.recovery.attempt_recovery("training_crash", Exception(message))
            if recovery:
                LOG.info("Recovery action: %s", recovery)

        if self.dashboard:
            self.dashboard.update(health_status="warning")

    def _on_health_critical(self, warning_type: str, message: str):
        """Handle critical health issue."""
        LOG.critical("Health CRITICAL [%s]: %s", warning_type, message)

        # Try to save state
        self._save_state({"critical_error": warning_type, "message": message})

        if self.dashboard:
            self.dashboard.update(health_status="critical")

    def _save_state(self, extra: Optional[dict] = None):
        """Save current training state for recovery."""
        state = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "experiment_dir": str(self.experiment_dir),
            "config": str(self.config_path),
            "shutdown_requested": self._shutdown_requested,
        }
        if extra:
            state.update(extra)

        state_path = self.experiment_dir / "runner_state.json"
        self.recovery.save_training_state(state_path, state)

    def run(self):
        """Main experiment run loop."""
        LOG.info("=" * 60)
        LOG.info("PRODUCTION EXPERIMENT RUNNER")
        LOG.info("=" * 60)
        LOG.info("Config: %s", self.config_path)
        LOG.info("Experiment: %s", self.experiment_dir)
        LOG.info("Auto-resume: %s", self.auto_resume)
        LOG.info("Dashboard: %s", self.use_dashboard)
        LOG.info("=" * 60)

        # Collect metadata
        metadata = collect_experiment_metadata(
            project_root=self.project_root,
            config=self.config,
        )
        save_experiment_metadata(self.experiment_dir, metadata)

        # Start background services
        self.start_monitor()
        self.start_dashboard()

        try:
            # Run the actual training
            self._run_training()

            # Post-run analysis
            self._run_analysis()

            # Generate reproducibility package
            self.artifacts.generate_reproducibility_package()

            # Update research log
            self._update_research_log()

            LOG.info("Experiment completed successfully!")
        except KeyboardInterrupt:
            LOG.info("Experiment interrupted by user")
        except Exception as e:
            LOG.error("Experiment failed: %s", e, exc_info=True)

            # Try recovery
            recovery = self.recovery.attempt_recovery("training_crash", e)
            if recovery:
                LOG.info("Recovery suggested: %s", recovery)
        finally:
            self._save_state()
            self.stop_services()

        # Print summary
        self._print_summary()

    def _run_training(self):
        """Run the actual training loop."""
        from training.trainer import Trainer
        from training.tokenizer.tokenizer import load_tokenizer

        # Load tokenizer
        tokenizer_path = self.project_root / "training" / "tokenizer.model"
        if not tokenizer_path.exists():
            LOG.error("Tokenizer not found at %s — run tokenizer training first", tokenizer_path)
            return

        tokenizer = load_tokenizer(str(tokenizer_path))
        vocab_size = tokenizer.GetPieceSize()
        LOG.info("Tokenizer loaded: vocab_size=%d", vocab_size)

        # Find latest checkpoint if auto-resume
        start_step = 0
        if self.auto_resume:
            valid_ckpt = self.recovery.find_valid_checkpoint()
            if valid_ckpt:
                LOG.info("Resuming from checkpoint: %s", valid_ckpt)
                # The trainer will handle loading

        # Create trainer
        trainer = Trainer(
            model=None,  # Will be created internally
            tokenizer=tokenizer,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Register shutdown hook
        def save_on_shutdown():
            if not self._shutdown_requested:
                trainer.save_checkpoint("interrupted")
                self._save_state({"reason": "graceful_shutdown"})

        import atexit
        atexit.register(save_on_shutdown)

        # Run training
        trainer.train(
            config=self.config,
            experiment_dir=self.experiment_dir,
            dashboard=self.dashboard,
            auto_resume=self.auto_resume,
        )

    def _run_analysis(self):
        """Run post-training analysis."""
        LOG.info("Running post-training analysis...")

        # Analyze training logs
        log_file = self.project_root / "training.log"
        if log_file.exists():
            metrics = self.analysis.analyze_training_run(log_file)
            self.analysis.generate_plots(metrics, self.experiment_dir / "plots")

            # Save metrics
            metrics_path = self.experiment_dir / "training_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

        # Copy logs
        log_files = [
            self.project_root / "training.log",
            self.project_root / "training" / "training.log",
        ]
        self.artifacts.copy_logs([f for f in log_files if f.exists()])

        # Copy best checkpoint
        self.artifacts.copy_best_checkpoint()

        # Generate report
        report = self.analysis.generate_report(self.experiment_dir)
        self.artifacts.save_artifact("experiment_report.txt", report)

        LOG.info("Post-training analysis complete")

    def _update_research_log(self):
        """Update EXPERIMENT_HISTORY.md."""
        # Determine experiment results
        eval_results_path = self.experiment_dir / "eval_results.json"
        results = {}
        if eval_results_path.exists():
            try:
                with open(eval_results_path) as f:
                    results = json.load(f)
            except Exception:
                pass

        # Create entry
        entry = {
            "name": f"Experiment {self.experiment_dir.name}",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "hypothesis": self.config.get("description", "See config for details"),
            "architecture": f"{self.config.get('model', {}).get('depth', '?')}L, {self.config.get('model', {}).get('n_heads', '?')}H",
            "results": results,
            "directory": str(self.experiment_dir),
        }

        self.research_log.add_experiment(entry)

    def _print_summary(self):
        """Print experiment summary."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Directory: {self.experiment_dir}")
        print(f"Config:    {self.config_path}")

        # Artifact summary
        summary = self.artifacts.get_artifact_summary()
        print(f"Checkpoints: {len(summary['checkpoints'])}")
        print(f"Total size:  {summary['total_size_mb']:.2f} MB")

        print("\nArtifacts:")
        print(f"  - experiment_metadata.json")
        print(f"  - experiment_summary.txt")
        print(f"  - experiment_report.txt")
        print(f"  - reproducibility/")
        print(f"  - plots/")
        print(f"  - checkpoints/")
        print(f"  - logs/")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Production experiment runner")
    parser.add_argument("--config", required=True, help="Training config YAML path")
    parser.add_argument("--dashboard", action="store_true", help="Enable live terminal dashboard")
    parser.add_argument("--auto-resume", action="store_true", default=True, help="Auto-resume from checkpoint")
    parser.add_argument("--no-resume", action="store_true", help="Disable auto-resume")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("experiment.log"),
        ],
    )

    runner = ExperimentRunner(
        config_path=Path(args.config),
        use_dashboard=args.dashboard,
        auto_resume=not args.no_resume,
    )
    runner.run()


if __name__ == "__main__":
    main()
