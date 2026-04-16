"""Conditional Feature Importance for Multi-Time Series Forecasting.

xeries provides tools for computing conditional feature importance metrics
targeted at time-series forecasting tasks. It supports multiple importance
algorithms and integrates with popular forecasting frameworks.

Key Classes:
    - ConditionalPermutationImportance: Permutation-based feature importance
    - ConditionalSHAP: SHAP-based feature importance with series-specific backgrounds
    - BatchSHAP: Efficient batch SHAP computation with external explainer support
    - TreePartitioner: Automatic subgroup discovery
    - ManualPartitioner: User-defined conditional subgroups
    - SklearnAdapter: scikit-learn model adapter
    - SkforecastAdapter: skforecast forecaster adapter
    - HierarchicalExplainer: Hierarchical aggregation of feature importance
    - HierarchyDefinition: Define hierarchical structure for multi-series data

Example:
    >>> from xeries import ConditionalPermutationImportance
    >>> from xeries.adapters import from_skforecast
    >>> adapter = from_skforecast(forecaster, series=data)
    >>> explainer = ConditionalPermutationImportance(model=adapter, metric='mse')
    >>> result = explainer.explain(X, y)

BatchSHAP Example (Efficient batch computation):
    >>> import shap
    >>> from xeries import BatchSHAP
    >>> tree_explainer = shap.TreeExplainer(model)
    >>> batch_shap = BatchSHAP(explainer=tree_explainer, series_col='level')
    >>> result = batch_shap.explain(X)

Hierarchical Example:
    >>> from xeries.hierarchy import HierarchyDefinition, HierarchicalExplainer
    >>> from xeries import ConditionalSHAP
    >>> hierarchy = HierarchyDefinition(
    ...     levels=["state", "store"],
    ...     columns=["state_id", "store_id"]
    ... )
    >>> base_explainer = ConditionalSHAP(model, X_train, series_col='level')
    >>> explainer = HierarchicalExplainer(base_explainer, hierarchy)
    >>> result = explainer.explain(X_test)
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
from xeries.hierarchy import (
    HierarchicalAggregator,
    HierarchicalExplainer,
    HierarchicalResult,
    HierarchyDefinition,
)
from xeries.importance import BatchSHAP, ConditionalPermutationImportance, ConditionalSHAP
from xeries.partitioners import ManualPartitioner, TreePartitioner
from xeries.visualization import (
    plot_hierarchy_bar,
    plot_hierarchy_comparison,
    plot_hierarchy_heatmap,
    plot_hierarchy_summary,
    plot_hierarchy_tree,
    plot_hierarchy_violin,
    plot_importance_bar,
    plot_importance_comparison,
    plot_importance_heatmap,
    plot_importance_per_series,
    plot_shap_bar,
    plot_shap_summary,
)

__all__ = [
    # Core types and base classes
    "ArrayLike",
    "BaseAdapter",
    "BaseExplainer",
    "BasePartitioner",
    # Importance methods
    "BatchSHAP",
    "ConditionalPermutationImportance",
    "ConditionalSHAP",
    "FeatureImportanceResult",
    "GroupLabels",
    # Hierarchical importance
    "HierarchicalAggregator",
    "HierarchicalExplainer",
    "HierarchicalResult",
    "HierarchyDefinition",
    # Partitioners
    "ManualPartitioner",
    "ModelProtocol",
    # Adapters
    "SkforecastAdapter",
    "SklearnAdapter",
    "TreePartitioner",
    # Version
    "__version__",
    "from_skforecast",
    # Standard visualization
    "plot_importance_bar",
    "plot_importance_comparison",
    "plot_importance_heatmap",
    "plot_importance_per_series",
    "plot_shap_bar",
    "plot_shap_summary",
    # Hierarchical visualization
    "plot_hierarchy_bar",
    "plot_hierarchy_comparison",
    "plot_hierarchy_heatmap",
    "plot_hierarchy_summary",
    "plot_hierarchy_tree",
    "plot_hierarchy_violin",
]
