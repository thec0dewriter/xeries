"""
xeries - eXplainability for Time Series

A Python library for conditional feature importance in multi-time series forecasting.
"""

from xeries._version import __version__
from xeries.adapters.darts import DartsAdapter
from xeries.adapters.skforecast import SkforecastAdapter, from_skforecast
from xeries.adapters.sklearn import SklearnAdapter
from xeries.dashboard.core import Dashboard
from xeries.core.types import CausalResult, RefutationResult
from xeries.importance.causal import CausalFeatureImportance
from xeries.importance.dropping import ConditionalDropImportance
from xeries.importance.permutation import ConditionalPermutationImportance
from xeries.importance.shap import ConditionalSHAP
from xeries.importance.shapiq import ConditionalSHAPIQ, SHAPIQResult
from xeries.partitioners.manual import ManualPartitioner
from xeries.partitioners.tree import TreePartitioner

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
