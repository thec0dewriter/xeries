"""Analysis helpers for temporal, statistical, and method comparison workflows."""

from xeries.analysis.comparison import compare_rankings
from xeries.analysis.error import ErrorAnalyzer
from xeries.analysis.significance import bootstrap_interval, estimate_significance
from xeries.analysis.temporal import TemporalImportance

__all__ = [
    "ErrorAnalyzer",
    "TemporalImportance",
    "bootstrap_interval",
    "compare_rankings",
    "estimate_significance",
]
