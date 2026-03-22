"""Base classes for tcpfi components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tcpfi.core.types import (
        ArrayLike,
        FeatureImportanceResult,
        GroupLabels,
        MetricFunction,
        ModelProtocol,
    )


class BasePartitioner(ABC):
    """Abstract base class for data partitioners.

    Partitioners create groups/subsets of data for conditional permutation.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, feature: str) -> BasePartitioner:
        """Fit the partitioner to the data.

        Args:
            X: Input features DataFrame.
            feature: The feature to condition on.

        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def get_groups(self, X: pd.DataFrame) -> NDArray[np.intp]:
        """Get group labels for each sample in X.

        Args:
            X: Input features DataFrame.

        Returns:
            Array of group labels with same length as X.
        """
        ...

    def fit_get_groups(self, X: pd.DataFrame, feature: str) -> NDArray[np.intp]:
        """Fit and return groups in one step.

        Args:
            X: Input features DataFrame.
            feature: The feature to condition on.

        Returns:
            Array of group labels.
        """
        self.fit(X, feature)
        return self.get_groups(X)


class BaseExplainer(ABC):
    """Abstract base class for feature importance explainers."""

    def __init__(
        self,
        model: ModelProtocol,
        metric: MetricFunction | str = "mse",
        random_state: int | None = None,
    ) -> None:
        """Initialize the explainer.

        Args:
            model: A model with a predict method.
            metric: Scoring metric ('mse', 'mae', 'rmse') or callable.
            random_state: Random seed for reproducibility.
        """
        self.model = model
        self.metric = self._resolve_metric(metric)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def _resolve_metric(self, metric: MetricFunction | str) -> MetricFunction:
        """Resolve metric string to callable."""
        if callable(metric):
            return metric  # type: ignore[return-value]

        metrics: dict[str, MetricFunction] = {
            "mse": self._mse,
            "mae": self._mae,
            "rmse": self._rmse,
        }
        if metric not in metrics:
            raise ValueError(f"Unknown metric: {metric}. Choose from {list(metrics.keys())}")
        return metrics[metric]

    @staticmethod
    def _mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Mean squared error."""
        return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

    @staticmethod
    def _mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Mean absolute error."""
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    @staticmethod
    def _rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Root mean squared error."""
        return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

    @abstractmethod
    def compute(
        self,
        X: pd.DataFrame,
        y: ArrayLike,
        features: list[str] | None = None,
        groups: GroupLabels | None = None,
    ) -> FeatureImportanceResult:
        """Compute feature importance.

        Args:
            X: Input features DataFrame.
            y: Target values.
            features: List of features to compute importance for.
            groups: Pre-defined group labels for conditional permutation.

        Returns:
            FeatureImportanceResult containing importance scores.
        """
        ...
