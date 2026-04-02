"""Adapter for sklearn-compatible estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from xeries.adapters.base import BaseAdapter


class SklearnAdapter(BaseAdapter):
    """Adapter for sklearn-like estimators with a predict method.

    This adapter standardizes plain tabular models for use with xeries
    explainers.
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        series_col: str = "series",
    ) -> None:
        """Initialize the adapter.

        Args:
            model: Fitted estimator implementing ``predict``.
            X_train: Training feature matrix.
            y_train: Training target vector.
            series_col: Column containing series identifiers.

        Raises:
            ValueError: If model does not expose ``predict``.
        """
        if not hasattr(model, "predict"):
            raise ValueError("model must implement predict(X)")

        self.model = model
        self._X = X_train.copy()
        self._y = pd.Series(y_train).copy()
        self._series_col = series_col

    def get_training_data(self, *args: Any, **kwargs: Any) -> tuple[pd.DataFrame, pd.Series]:
        """Return the training data provided at construction time."""
        return self._X, self._y

    def predict(self, X: pd.DataFrame) -> NDArray[Any]:
        """Predict with the underlying sklearn-like estimator."""
        return np.asarray(self.model.predict(X))

    def get_feature_names(self) -> list[str]:
        """Return feature column names."""
        return list(self._X.columns)

    def get_series_column(self) -> str:
        """Return the configured series identifier column name."""
        return self._series_col
