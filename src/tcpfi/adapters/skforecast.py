"""Adapter for skforecast ForecasterMultiSeries integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tcpfi.adapters.base import BaseAdapter

if TYPE_CHECKING:
    pass


class SkforecastAdapter(BaseAdapter):
    """Adapter for skforecast ForecasterMultiSeries.

    Provides integration with skforecast's multi-series forecasting models,
    extracting the training matrix and providing prediction capabilities
    for use with tcpfi explainers.

    Example:
        >>> from skforecast.ForecasterMultiSeries import ForecasterMultiSeries
        >>> forecaster = ForecasterMultiSeries(regressor, lags=24)
        >>> forecaster.fit(series_data)
        >>>
        >>> adapter = SkforecastAdapter(forecaster)
        >>> X, y = adapter.get_training_data()
        >>> explainer = ConditionalPermutationImportance(adapter)
        >>> result = explainer.compute(X, y)
    """

    SERIES_COL = "level"

    def __init__(self, forecaster: Any) -> None:
        """Initialize the skforecast adapter.

        Args:
            forecaster: A fitted skforecast ForecasterMultiSeries instance.

        Raises:
            ValueError: If forecaster is not fitted.
        """
        self._validate_forecaster(forecaster)
        self.forecaster = forecaster
        self._X: pd.DataFrame | None = None
        self._y: pd.Series | None = None

    def _validate_forecaster(self, forecaster: Any) -> None:
        """Validate that the forecaster is properly configured."""
        if not hasattr(forecaster, "regressor"):
            raise ValueError(
                "forecaster must be a skforecast ForecasterMultiSeries instance"
            )

        if not hasattr(forecaster, "is_fitted") or not forecaster.is_fitted:
            raise ValueError("forecaster must be fitted before creating an adapter")

    def get_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Extract training features (X) and target (y) from the forecaster.

        The training matrix X has a MultiIndex with 'level' (series_id) and 'date'.

        Returns:
            Tuple of (X, y) DataFrames.
        """
        if self._X is None or self._y is None:
            self._X, self._y = self.forecaster.create_train_X_y()

        return self._X, self._y

    def predict(self, X: pd.DataFrame) -> NDArray[Any]:
        """Make predictions using the underlying regressor.

        Args:
            X: Input features DataFrame (same structure as training X).

        Returns:
            Array of predictions.
        """
        X_values = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return np.asarray(self.forecaster.regressor.predict(X_values))

    def get_feature_names(self) -> list[str]:
        """Get the names of features (lag columns, etc.).

        Returns:
            List of feature names from the training matrix.
        """
        X, _ = self.get_training_data()
        return list(X.columns)

    def get_series_column(self) -> str:
        """Get the name of the series identifier column.

        Returns:
            The series column name ('level' for skforecast).
        """
        return self.SERIES_COL

    def get_series_ids(self) -> list[Any]:
        """Get unique series identifiers.

        Returns:
            List of unique series IDs.
        """
        X, _ = self.get_training_data()
        return list(X.index.get_level_values(self.SERIES_COL).unique())

    def get_lag_features(self) -> list[str]:
        """Get the names of lag features.

        Returns:
            List of lag feature names (e.g., ['lag_1', 'lag_2', ...]).
        """
        feature_names = self.get_feature_names()
        return [f for f in feature_names if f.startswith("lag_")]

    @property
    def n_lags(self) -> int:
        """Return the number of lags used by the forecaster."""
        return len(self.get_lag_features())


def from_skforecast(forecaster: Any) -> SkforecastAdapter:
    """Create a SkforecastAdapter from a fitted ForecasterMultiSeries.

    Convenience function for creating adapters.

    Args:
        forecaster: A fitted skforecast ForecasterMultiSeries instance.

    Returns:
        SkforecastAdapter instance.
    """
    return SkforecastAdapter(forecaster)
