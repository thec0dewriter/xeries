"""Conditional Permutation Feature Importance implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray

from timelens.core.base import BasePartitioner, MetricBasedExplainer
from timelens.core.types import (
    ArrayLike,
    FeatureImportanceResult,
    GroupLabels,
    MetricFunction,
    ModelProtocol,
)
from timelens.partitioners.tree import TreePartitioner

if TYPE_CHECKING:
    pass


class ConditionalPermutationImportance(MetricBasedExplainer):
    """Conditional Permutation Feature Importance calculator.

    This implements conditional permutation importance where feature values
    are only shuffled within defined subgroups, preserving the correlation
    structure between features.

    Supports two strategies:
    - 'auto': Uses tree-based cs-PFI to automatically learn subgroups
    - 'manual': Uses pre-defined groups provided by the user

    Example:
        >>> explainer = ConditionalPermutationImportance(model, metric='mse')
        >>> result = explainer.explain(X, y, features=['lag_1', 'lag_2'])
        >>> print(result.to_dataframe())
    """

    def __init__(
        self,
        model: ModelProtocol,
        metric: MetricFunction | str = "mse",
        strategy: str = "auto",
        partitioner: BasePartitioner | None = None,
        n_repeats: int = 5,
        n_jobs: int = -1,
        random_state: int | None = None,
    ) -> None:
        """Initialize the conditional permutation importance calculator.

        Args:
            model: A model with a predict method.
            metric: Scoring metric ('mse', 'mae', 'rmse') or callable.
            strategy: Grouping strategy ('auto' for tree-based, 'manual' for user-defined).
            partitioner: Custom partitioner instance. If None, uses TreePartitioner for 'auto'.
            n_repeats: Number of times to repeat permutation for each feature.
            n_jobs: Number of parallel jobs (-1 for all cores).
            random_state: Random seed for reproducibility.
        """
        super().__init__(model, metric, random_state)
        self.strategy = strategy
        self.partitioner = partitioner
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs

    def explain(
        self,
        X: pd.DataFrame,
        y: ArrayLike,
        features: list[str] | None = None,
        groups: GroupLabels | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> FeatureImportanceResult:  # type: ignore[override]
        """Compute conditional permutation importance for features.

        Args:
            X: Input features DataFrame.
            y: Target values.
            features: List of features to compute importance for.
                If None, uses all columns in X.
            groups: Pre-defined group labels for 'manual' strategy.
                Required when strategy='manual' and no partitioner is provided.

        Returns:
            FeatureImportanceResult containing importance scores.
        """
        y_array = np.asarray(y)
        features = features or list(X.columns)

        baseline_pred = self.model.predict(X)
        baseline_score = self.metric(y_array, baseline_pred)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_feature_importance)(X, y_array, feature, baseline_score, groups)
            for feature in features
        )

        importances = []
        stds = []
        permuted_scores: dict[str, list[float]] = {}

        for feature, scores in zip(features, results, strict=True):
            importance_values = [score - baseline_score for score in scores]
            importances.append(np.mean(importance_values))
            stds.append(np.std(importance_values))
            permuted_scores[feature] = scores

        return FeatureImportanceResult(
            feature_names=features,
            importances=np.array(importances),
            std=np.array(stds),
            baseline_score=baseline_score,
            permuted_scores=permuted_scores,
            method="conditional_permutation",
            n_repeats=self.n_repeats,
        )

    def _compute_feature_importance(
        self,
        X: pd.DataFrame,
        y: NDArray[np.floating[Any]],
        feature: str,
        baseline_score: float,
        groups: GroupLabels | None,
    ) -> list[float]:
        """Compute importance for a single feature across repeats."""
        group_labels = self._get_groups(X, feature, groups)

        scores = []
        for repeat in range(self.n_repeats):
            rng = np.random.default_rng(self.random_state + repeat if self.random_state else None)
            X_permuted = self._conditional_permute(X, feature, group_labels, rng)
            permuted_pred = self.model.predict(X_permuted)
            score = self.metric(y, permuted_pred)
            scores.append(score)

        return scores

    def _get_groups(
        self,
        X: pd.DataFrame,
        feature: str,
        groups: GroupLabels | None,
    ) -> NDArray[np.intp]:
        """Get group labels for conditional permutation."""
        if groups is not None:
            return np.asarray(groups).astype(np.intp)

        if self.partitioner is not None:
            return self.partitioner.fit_get_groups(X, feature)

        if self.strategy == "auto":
            partitioner = TreePartitioner(random_state=self.random_state)
            return partitioner.fit_get_groups(X, feature)

        raise ValueError("For strategy='manual', either provide 'groups' or a 'partitioner'")

    def _conditional_permute(
        self,
        X: pd.DataFrame,
        feature: str,
        groups: NDArray[np.intp],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Permute feature values within groups.

        Args:
            X: Input DataFrame.
            feature: Feature column to permute.
            groups: Group labels for each row.
            rng: Random number generator.

        Returns:
            DataFrame with permuted feature values.
        """
        X_permuted = X.copy()
        feature_values = X_permuted[feature].to_numpy()

        unique_groups = np.unique(groups)
        permuted_values = feature_values.copy()

        for group in unique_groups:
            mask = groups == group
            group_values = feature_values[mask]
            permuted_values[mask] = rng.permutation(group_values)

        X_permuted[feature] = permuted_values
        return X_permuted
