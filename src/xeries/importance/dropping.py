"""Conditional Feature Dropping Importance implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from xeries.core.base import BasePartitioner, MetricBasedExplainer
from xeries.core.types import (
    ArrayLike,
    FeatureImportanceResult,
    GroupLabels,
    MetricFunction,
    ModelProtocol,
)
from xeries.partitioners.tree import TreePartitioner

if TYPE_CHECKING:
    pass


class ConditionalDropImportance(MetricBasedExplainer):
    """Conditional Feature Dropping Importance calculator.

    Instead of permuting a feature's values, this method replaces them with
    a fill value (mean, median, zero, or noise) computed *within* each
    conditional subgroup.  Measuring the resulting performance drop isolates
    each feature's contribution while respecting the group structure.

    Supports two strategies:
    - 'auto': Uses tree-based cs-PFI to automatically learn subgroups
    - 'manual': Uses pre-defined groups provided by the user

    Example:
        >>> explainer = ConditionalDropImportance(model, metric='mse', fill_strategy='mean')
        >>> result = explainer.explain(X, y, features=['lag_1', 'lag_2'])
        >>> print(result.to_dataframe())
    """

    def __init__(
        self,
        model: ModelProtocol,
        metric: MetricFunction | str = "mse",
        fill_strategy: str = "mean",
        strategy: str = "auto",
        partitioner: BasePartitioner | None = None,
        n_jobs: int = -1,
        random_state: int | None = None,
    ) -> None:
        """Initialize the conditional drop importance calculator.

        Args:
            model: A model with a predict method.
            metric: Scoring metric ('mse', 'mae', 'rmse') or callable.
            fill_strategy: How to replace dropped feature values.
                Options: 'mean', 'median', 'zero', 'noise'.
            strategy: Grouping strategy ('auto' for tree-based, 'manual' for user-defined).
            partitioner: Custom partitioner instance. If None, uses TreePartitioner for 'auto'.
            n_jobs: Number of parallel jobs (-1 for all cores). Reserved for future use.
            random_state: Random seed for reproducibility.
        """
        super().__init__(model, metric, random_state)
        if fill_strategy not in ("mean", "median", "zero", "noise"):
            raise ValueError(
                f"Unknown fill_strategy: {fill_strategy}. "
                "Choose from 'mean', 'median', 'zero', 'noise'."
            )
        self.fill_strategy = fill_strategy
        self.strategy = strategy
        self.partitioner = partitioner
        self.n_jobs = n_jobs

    def explain(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        y: ArrayLike,
        features: list[str] | None = None,
        groups: GroupLabels | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> FeatureImportanceResult:
        """Compute conditional drop importance for features.

        Args:
            X: Input features DataFrame.
            y: Target values.
            features: List of features to compute importance for.
                If None, uses all columns in X.
            groups: Pre-defined group labels for 'manual' strategy.

        Returns:
            FeatureImportanceResult containing importance scores.
        """
        y_array = np.asarray(y)
        features = features or list(X.columns)

        baseline_pred = self.model.predict(X)
        baseline_score = self.metric(y_array, baseline_pred)

        importances: list[float] = []

        for feature in features:
            group_labels = self._get_groups(X, feature, groups)
            X_dropped = self._fill_feature(X, feature, group_labels)
            dropped_pred = self.model.predict(X_dropped)
            dropped_score = self.metric(y_array, dropped_pred)
            importances.append(dropped_score - baseline_score)

        return FeatureImportanceResult(
            feature_names=features,
            importances=np.array(importances),
            std=None,
            baseline_score=baseline_score,
            method="conditional_drop",
            n_repeats=1,
        )

    def _get_groups(
        self,
        X: pd.DataFrame,
        feature: str,
        groups: GroupLabels | None,
    ) -> NDArray[np.intp]:
        """Get group labels for conditional filling."""
        if groups is not None:
            return np.asarray(groups).astype(np.intp)

        if self.partitioner is not None:
            return self.partitioner.fit_get_groups(X, feature)

        if self.strategy == "auto":
            partitioner = TreePartitioner(random_state=self.random_state)
            return partitioner.fit_get_groups(X, feature)

        raise ValueError("For strategy='manual', either provide 'groups' or a 'partitioner'")

    def _fill_feature(
        self,
        X: pd.DataFrame,
        feature: str,
        groups: NDArray[np.intp],
    ) -> pd.DataFrame:
        """Replace feature values within groups using the fill strategy.

        Args:
            X: Input DataFrame.
            feature: Feature column to fill.
            groups: Group labels for each row.

        Returns:
            DataFrame with replaced feature values.
        """
        X_filled = X.copy()
        feature_values = X_filled[feature].to_numpy(dtype=np.float64)
        filled_values = feature_values.copy()

        unique_groups = np.unique(groups)

        for group in unique_groups:
            mask = groups == group
            group_vals = feature_values[mask]

            if self.fill_strategy == "mean":
                fill_val = np.mean(group_vals)
                filled_values[mask] = fill_val
            elif self.fill_strategy == "median":
                fill_val = np.median(group_vals)
                filled_values[mask] = fill_val
            elif self.fill_strategy == "zero":
                filled_values[mask] = 0.0
            elif self.fill_strategy == "noise":
                rng = np.random.default_rng(self.random_state)
                filled_values[mask] = rng.normal(
                    loc=np.mean(group_vals),
                    scale=max(np.std(group_vals), 1e-8),
                    size=int(np.sum(mask)),
                )

        X_filled[feature] = filled_values
        return X_filled
