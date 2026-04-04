"""Base classes for xeries components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from xeries.core.types import (
        ArrayLike,
        BaseResult,
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
    """Abstract base class for all feature explainers."""

    @abstractmethod
    def explain(
        self,
        X: pd.DataFrame,
        *args: Any,
        **kwargs: Any,
    ) -> BaseResult:
        """Compute the explanation.

        Args:
            X: Input features DataFrame.
            *args: Additional arguments required by specific explainers.
            **kwargs: Additional keyword arguments required by specific explainers.

        Returns:
            The explanation result.
        """
        ...


class MetricBasedExplainer(BaseExplainer):
    """Base class for explainers that rely on predictive performance metrics.

    This includes permutation feature importance and other dropping methods.
    """

    def __init__(
        self,
        model: ModelProtocol,
        metric: MetricFunction | str = "mse",
        random_state: int | None = None,
    ) -> None:
        """Initialize the metric-based explainer.

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
        if not isinstance(metric, str):
            return metric

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


class AttributionExplainer(BaseExplainer):
    """Base class for explainers that attribute predictions directly to features.

    This includes SHAP, SHAP-IQ, and other local attribution methods.
    """

    def __init__(
        self,
        model: ModelProtocol,
        background_data: pd.DataFrame,
        random_state: int | None = None,
    ) -> None:
        """Initialize the attribution explainer.

        Args:
            model: A model with a predict method.
            background_data: Dataset to draw background samples from.
            random_state: Random seed for reproducibility.
        """
        self.model = model
        self.background_data = background_data
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)


class CausalExplainer(BaseExplainer):
    """Base class for causal inference based explainers.

    This serves as an integration point for structural causal models
    and DAG-based conditional feature importance. Subclasses implement
    the explain() method using causal estimators (e.g. DoWhy/EconML).
    """

    def __init__(
        self,
        model: ModelProtocol,
        treatment_features: list[str],
        causal_graph: Any | None = None,
        series_col: str = "level",
        random_state: int | None = None,
    ) -> None:
        """Initialize the causal explainer.

        Args:
            model: A model with a predict method.
            treatment_features: Features to estimate causal effects for.
            causal_graph: Optional DAG (networkx DiGraph or DoWhy graph string).
            series_col: Column or index level containing series identifiers.
            random_state: Random seed for reproducibility.
        """
        self.model = model
        self.treatment_features = treatment_features
        self.causal_graph = causal_graph
        self.series_col = series_col
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
