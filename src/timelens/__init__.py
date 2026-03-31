"""
timelens - Time-Conditional Permutation Feature Importance

A Python library for conditional feature importance in multi-time series forecasting.
"""

from timelens._version import __version__
from timelens.importance.permutation import ConditionalPermutationImportance
from timelens.importance.shap import ConditionalSHAP
from timelens.partitioners.manual import ManualPartitioner
from timelens.partitioners.tree import TreePartitioner

__all__ = [
    "ConditionalPermutationImportance",
    "ConditionalSHAP",
    "ManualPartitioner",
    "TreePartitioner",
    "__version__",
]
