"""Unified logging system supporting TensorBoard, CSV, and W&B.

All loggers implement a common interface:
    - log_scalar(tag, value, step)
    - log_scalars(tag, values_dict, step)
    - close()
"""

import os
import csv
import time
from typing import Dict, Any, Optional


class CSVLogger:
    """CSV logger for training metrics."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = None

    def log_scalar(self, tag: str, value: float, step: int):
        if self._writer is None:
            self._writer = csv.writer(self._file)
            self._writer.writerow(["step", "tag", "value", "timestamp"])
        self._writer.writerow([step, tag, value, time.time()])
        self._file.flush()

    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        for k, v in values.items():
            self.log_scalar(f"{tag}/{k}", v, step)

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()


class TensorBoardLogger:
    """TensorBoard logger wrapper."""

    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            print("TensorBoard not available (pip install tensorboard)")
            self.writer = None

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        if self.writer:
            self.writer.add_scalars(tag, values, step)

    def log_histogram(self, tag: str, values, step: int):
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def close(self):
        if self.writer:
            self.writer.close()


class WandBLogger:
    """Weights & Biases logger (optional)."""

    def __init__(self, project: str, config: Optional[Dict] = None, name: Optional[str] = None):
        try:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=project, config=config, name=name)
        except ImportError:
            print("W&B not available (pip install wandb)")
            self.wandb = None

    def log_scalar(self, tag: str, value: float, step: int):
        if self.wandb:
            self.wandb.log({tag: value}, step=step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        if self.wandb:
            log_dict = {f"{tag}/{k}": v for k, v in values.items()}
            log_dict["train/step"] = step
            self.wandb.log(log_dict)

    def log_config(self, config: Dict):
        if self.wandb:
            self.wandb.config.update(config)

    def close(self):
        if self.wandb:
            self.wandb.finish()


class LoggerManager:
    """Unified logger that dispatches to multiple backends.

    Args:
        log_dir: Base log directory.
        tensorboard: Enable TensorBoard.
        csv: Enable CSV logging.
        wandb: Enable W&B logging.
        wandb_project: W&B project name.
        config: Config dict to log to W&B.
    """

    def __init__(
        self,
        log_dir: str = "training/logs",
        tensorboard: bool = True,
        csv: bool = True,
        wandb: bool = False,
        wandb_project: str = "deepseek-baseline",
        config: Optional[Dict] = None,
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.loggers = []

        if tensorboard:
            self.loggers.append(TensorBoardLogger(os.path.join(log_dir, "tensorboard")))

        if csv:
            self.loggers.append(CSVLogger(os.path.join(log_dir, "metrics.csv")))

        if wandb:
            self.loggers.append(WandBLogger(project=wandb_project, config=config))

    def log_scalar(self, tag: str, value: float, step: int):
        for logger in self.loggers:
            logger.log_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        for logger in self.loggers:
            logger.log_scalars(tag, values, step)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log multiple metrics at once."""
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int):
        for logger in self.loggers:
            if hasattr(logger, "log_histogram"):
                logger.log_histogram(tag, values, step)

    def close(self):
        for logger in self.loggers:
            logger.close()
