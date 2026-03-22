"""
tcpfi - Time-Conditional Permutation Feature Importance

A Python library for conditional feature importance in multi-time series forecasting.
"""

from tcpfi._version import __version__
from tcpfi.importance.permutation import ConditionalPermutationImportance
from tcpfi.importance.shap import ConditionalSHAP
from tcpfi.partitioners.manual import ManualPartitioner
from tcpfi.partitioners.tree import TreePartitioner

__all__ = [
    "ConditionalPermutationImportance",
    "ConditionalSHAP",
    "ManualPartitioner",
    "TreePartitioner",
    "__version__",
]
