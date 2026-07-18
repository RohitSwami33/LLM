"""Experiment tracking and reproducibility framework.

Every training run creates a unique experiment directory with full metadata,
enabling exact reproducibility and cross-run comparison.
"""

from .manager import ExperimentManager
from .system_info import SystemInfo
from .reports import ReportGenerator

__all__ = ["ExperimentManager", "SystemInfo", "ReportGenerator"]
