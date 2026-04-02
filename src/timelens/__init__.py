"""
timelens - Time-Conditional Permutation Feature Importance

A Python library for conditional feature importance in multi-time series forecasting.
"""

from timelens._version import __version__
from timelens.adapters.darts import DartsAdapter
from timelens.adapters.skforecast import SkforecastAdapter, from_skforecast
from timelens.adapters.sklearn import SklearnAdapter
from timelens.dashboard.core import Dashboard
from timelens.core.types import CausalResult, RefutationResult
from timelens.importance.causal import CausalFeatureImportance
from timelens.importance.dropping import ConditionalDropImportance
from timelens.importance.permutation import ConditionalPermutationImportance
from timelens.importance.shap import ConditionalSHAP
from timelens.importance.shapiq import ConditionalSHAPIQ, SHAPIQResult
from timelens.partitioners.manual import ManualPartitioner
from timelens.partitioners.tree import TreePartitioner

__all__ = [
    "CausalFeatureImportance",
    "CausalResult",
    "ConditionalDropImportance",
    "ConditionalPermutationImportance",
    "ConditionalSHAP",
    "ConditionalSHAPIQ",
    "Dashboard",
    "DartsAdapter",
    "ManualPartitioner",
    "RefutationResult",
    "SkforecastAdapter",
    "SklearnAdapter",
    "SHAPIQResult",
    "TreePartitioner",
    "__version__",
    "from_skforecast",
]
