"""Adapter for skforecast ForecasterRecursiveMultiSeries integration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from xeries.adapters.base import BaseAdapter


class SkforecastAdapter(BaseAdapter):
    """Adapter for skforecast :class:`~skforecast.recursive.ForecasterRecursiveMultiSeries`.

    In skforecast **0.21+**, the multi-series global forecaster was renamed from
    ``ForecasterAutoregMultiSeries`` / ``ForecasterMultiSeries`` to
    ``ForecasterRecursiveMultiSeries``, and ``create_train_X_y`` requires the same
    ``series`` object you passed to :meth:`fit`.

    Provides integration with skforecast's multi-series forecasting models,
    extracting the training matrix and providing prediction capabilities
    for use with xeries explainers.

    Example:
        >>> from skforecast.recursive import ForecasterRecursiveMultiSeries
        >>> from sklearn.ensemble import RandomForestRegressor
        >>>
        >>> forecaster = ForecasterRecursiveMultiSeries(
        ...     estimator=RandomForestRegressor(random_state=0),
        ...     lags=24,
        ... )
        >>> forecaster.fit(series=series_wide_df)
        >>>
        >>> adapter = SkforecastAdapter(forecaster, series=series_wide_df)
        >>> X, y = adapter.get_training_data()
        >>> # or: adapter = from_skforecast(forecaster, series=series_wide_df)
    """

    #: Legacy MultiIndex level name (older skforecast long / stacked matrices).
    SERIES_LEVEL_LEGACY = "level"
    #: Column name used by skforecast 0.21+ for ordinal series encoding (wide / dict input).
    SERIES_COL_ENCODED = "_level_skforecast"

    def __init__(
        self,
        forecaster: Any,
        series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
        exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
    ) -> None:
        """Initialize the skforecast adapter.

        Args:
            forecaster: A fitted ``ForecasterRecursiveMultiSeries`` instance.
            series: Training series in the same form as passed to ``fit(series=...)``.
                If provided, :meth:`get_training_data` can be called with no arguments.
            exog: Optional exogenous variables, same as passed to ``fit`` if any.

        Raises:
            ValueError: If forecaster is not fitted or is not a supported type.
        """
        self._validate_forecaster(forecaster)
        self.forecaster = forecaster
        self._series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = series
        self._exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = exog
        self._X: pd.DataFrame | None = None
        self._y: pd.Series | None = None
        self._cache_key: tuple[int | None, int | None] | None = None

    def get_series_column(self) -> str:
        """Column or index level name used to identify the series in ``X``.

        skforecast 0.21+ typically adds ``_level_skforecast`` (ordinal codes).
        Older stacked layouts use a MultiIndex level named ``level``.
        """
        X, _ = self.get_training_data()
        if isinstance(X.index, pd.MultiIndex) and self.SERIES_LEVEL_LEGACY in X.index.names:
            return self.SERIES_LEVEL_LEGACY
        if self.SERIES_COL_ENCODED in X.columns:
            return self.SERIES_COL_ENCODED
        if self.SERIES_LEVEL_LEGACY in X.columns:
            return self.SERIES_LEVEL_LEGACY
        raise ValueError(
            "Cannot infer series column: expected MultiIndex level 'level' or "
            f"column '{self.SERIES_COL_ENCODED}' in the training matrix."
        )

    def get_series_ids(self) -> list[Any]:
        """Unique series identifiers (decoded names when using ordinal encoding)."""
        X, _ = self.get_training_data()
        col = self.get_series_column()
        if col == self.SERIES_COL_ENCODED:
            codes = X[col]
            names = getattr(self.forecaster, "series_names_in_", None)
            if names is not None:
                return [names[int(i)] for i in sorted(codes.unique())]
            return sorted(codes.unique().tolist())
        if isinstance(X.index, pd.MultiIndex):
            return list(X.index.get_level_values(col).unique())
        return list(X[col].unique())

    @staticmethod
    def _validate_forecaster(forecaster: Any) -> None:
        """Validate that the forecaster is properly configured."""
        has_model = hasattr(forecaster, "estimator") or hasattr(forecaster, "regressor")
        if not has_model:
            raise ValueError(
                "forecaster must be a skforecast ForecasterRecursiveMultiSeries "
                "(or compatible) instance with an estimator/regressor"
            )

        if not hasattr(forecaster, "is_fitted") or not forecaster.is_fitted:
            raise ValueError("forecaster must be fitted before creating an adapter")

        if not hasattr(forecaster, "create_train_X_y"):
            raise ValueError("forecaster must implement create_train_X_y(series, ...)")

    @property
    def _fitted_estimator(self) -> Any:
        """Return the sklearn estimator (prefer ``estimator`` over deprecated ``regressor``)."""
        est = getattr(self.forecaster, "estimator", None)
        if est is not None:
            return est
        return getattr(self.forecaster, "regressor", None)

    def get_training_data(
        self,
        series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
        exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
        *,
        suppress_warnings: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Extract training features (X) and target (y) from the forecaster.

        skforecast's ``create_train_X_y`` requires the ``series`` argument (and optional
        ``exog``) matching what was used in ``fit``. Pass ``series`` here or when
        constructing :class:`SkforecastAdapter`.

        The training matrix X has a MultiIndex with ``level`` (series id) and ``date``.

        Args:
            series: Same ``series`` as ``forecaster.fit(series=...)``. Uses the
                constructor ``series`` / ``exog`` if omitted.
            exog: Same optional exog as in ``fit``, if used.
            suppress_warnings: Forwarded to skforecast.

        Returns:
            Tuple of ``(X, y)``.

        Raises:
            ValueError: If ``series`` cannot be resolved.
        """
        s = series if series is not None else self._series
        e = exog if exog is not None else self._exog
        if s is None:
            raise ValueError(
                "Pass `series` (the same object passed to forecaster.fit(series=...)) "
                "to get_training_data(), or provide series=... when constructing "
                "SkforecastAdapter(forecaster, series=...)."
            )

        key = (id(s), id(e) if e is not None else None)
        if self._cache_key != key or self._X is None or self._y is None:
            self._X, self._y = self.forecaster.create_train_X_y(
                s,
                exog=e,
                suppress_warnings=suppress_warnings,
            )
            self._cache_key = key

        self._series = s
        self._exog = e
        return self._X, self._y

    def predict(self, X: pd.DataFrame) -> NDArray[Any]:
        """Make predictions using the underlying estimator.

        Args:
            X: Input features DataFrame (same structure as training X).

        Returns:
            Array of predictions.
        """
        model = self._fitted_estimator
        if model is None:
            raise ValueError("No fitted estimator found on forecaster")
        X_values = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return np.asarray(model.predict(X_values))

    def get_feature_names(self) -> list[str]:
        """Get predictor column names (lags, window features, etc.)."""
        X, _ = self.get_training_data()
        return [c for c in X.columns if c != self.SERIES_COL_ENCODED]

    def get_lag_features(self) -> list[str]:
        """Get lag feature column names (e.g. ``lag_1``, ``lag_2``, ...)."""
        feature_names = self.get_feature_names()
        return [f for f in feature_names if f.startswith("lag_")]

    @property
    def n_lags(self) -> int:
        """Return the number of lag features in the training matrix."""
        return len(self.get_lag_features())


def from_skforecast(
    forecaster: Any,
    series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
    exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
) -> SkforecastAdapter:
    """Create a :class:`SkforecastAdapter` from a fitted forecaster.

    Args:
        forecaster: Fitted ``ForecasterRecursiveMultiSeries``.
        series: Training series (same as ``fit(series=...)``) if not passed later.
        exog: Optional exog matching ``fit``.

    Returns:
        SkforecastAdapter instance.
    """
    return SkforecastAdapter(forecaster, series=series, exog=exog)
