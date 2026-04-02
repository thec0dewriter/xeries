"""Analysis helpers for temporal, statistical, and method comparison workflows."""

from timelens.analysis.comparison import compare_rankings
from timelens.analysis.error import ErrorAnalyzer
from timelens.analysis.significance import bootstrap_interval, estimate_significance
from timelens.analysis.temporal import TemporalImportance

__all__ = [
    "ErrorAnalyzer",
    "TemporalImportance",
    "bootstrap_interval",
    "compare_rankings",
    "estimate_significance",
]
