"""Type definitions for tcpfi."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

ArrayLike = np.ndarray | pd.Series | pd.DataFrame
GroupLabels = np.ndarray | pd.Series | list[Any]


class ModelProtocol(Protocol):
    """Protocol for models that can be used with tcpfi explainers."""

    def predict(
        self, X: ArrayLike | pd.DataFrame
    ) -> NDArray[np.floating[Any]] | np.ndarray | pd.Series:
        """Make predictions on input data."""
        ...


@dataclass
class FeatureImportanceResult:
    """Container for feature importance results.

    Attributes:
        feature_names: List of feature names.
        importances: Array of importance scores for each feature.
        std: Standard deviations of importance scores (from multiple permutations).
        baseline_score: The baseline model score before permutation.
        permuted_scores: Dictionary mapping feature names to their permuted scores.
        method: The method used to compute importance ('permutation', 'shap', etc.).
        n_repeats: Number of permutation repeats used.
    """

    feature_names: list[str]
    importances: NDArray[np.floating[Any]]
    std: NDArray[np.floating[Any]] | None = None
    baseline_score: float = 0.0
    permuted_scores: dict[str, list[float]] = field(default_factory=dict)
    method: str = "permutation"
    n_repeats: int = 1

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        data = {
            "feature": self.feature_names,
            "importance": self.importances,
        }
        if self.std is not None:
            data["std"] = self.std
        return pd.DataFrame(data).sort_values("importance", ascending=False)


@dataclass
class SHAPResult:
    """Container for SHAP explanation results.

    Attributes:
        shap_values: SHAP values array with shape (n_samples, n_features).
        base_values: Base/expected values for each sample.
        feature_names: List of feature names.
        data: The input data used for explanation.
    """

    shap_values: NDArray[np.floating[Any]]
    base_values: NDArray[np.floating[Any]]
    feature_names: list[str]
    data: ArrayLike

    def mean_abs_shap(self) -> pd.DataFrame:
        """Compute mean absolute SHAP values per feature."""
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        return pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)


# Type alias for metric functions that take true and predicted values and return a score.
MetricFunction: TypeAlias = Callable[[ArrayLike, ArrayLike], float | int]
