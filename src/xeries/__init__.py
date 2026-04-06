"""Conditional Feature Importance for Multi-Time Series Forecasting.

xeries provides tools for computing conditional feature importance metrics
targeted at time-series forecasting tasks. It supports multiple importance
algorithms and integrates with popular forecasting frameworks.

Key Classes:
    - ConditionalPermutationImportance: Permutation-based feature importance
    - ConditionalSHAP: SHAP-based feature importance
    - TreePartitioner: Automatic subgroup discovery
    - ManualPartitioner: User-defined conditional subgroups
    - SklearnAdapter: scikit-learn model adapter
    - SkforecastAdapter: skforecast forecaster adapter

Example:
    >>> from xeries import ConditionalPermutationImportance
    >>> from xeries.adapters import from_skforecast
    >>> adapter = from_skforecast(forecaster, series=data)
    >>> explainer = ConditionalPermutationImportance(model=adapter, metric='mse')
    >>> result = explainer.explain(X, y)
"""

from __future__ import annotations

from xeries._version import __version__
from xeries.adapters import (
    BaseAdapter,
    SkforecastAdapter,
    SklearnAdapter,
    from_skforecast,
)
from xeries.core import (
    ArrayLike,
    BaseExplainer,
    BasePartitioner,
    FeatureImportanceResult,
    GroupLabels,
    ModelProtocol,
)
from xeries.importance import ConditionalPermutationImportance, ConditionalSHAP
from xeries.partitioners import ManualPartitioner, TreePartitioner
from xeries.visualization import (
    plot_importance_bar,
    plot_importance_comparison,
    plot_importance_heatmap,
    plot_importance_per_series,
    plot_shap_bar,
    plot_shap_summary,
)

__all__ = [
    # Version
    "__version__",
    # Core types and base classes
    "ArrayLike",
    "BaseAdapter",
    "BaseExplainer",
    "BasePartitioner",
    "FeatureImportanceResult",
    "GroupLabels",
    "ModelProtocol",
    # Importance methods
    "ConditionalPermutationImportance",
    "ConditionalSHAP",
    # Partitioners
    "ManualPartitioner",
    "TreePartitioner",
    # Adapters
    "SkforecastAdapter",
    "SklearnAdapter",
    "from_skforecast",
    # Visualization
    "plot_importance_bar",
    "plot_importance_comparison",
    "plot_importance_heatmap",
    "plot_importance_per_series",
    "plot_shap_bar",
    "plot_shap_summary",
]
