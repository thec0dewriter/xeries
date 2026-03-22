"""Shared test fixtures and utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor


class MockModel:
    """Mock model for testing that mimics sklearn interface."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self._fitted = False
        self._coef: NDArray[np.floating[Any]] | None = None

    def fit(self, X: pd.DataFrame | NDArray[Any], y: NDArray[Any]) -> MockModel:
        n_features = X.shape[1] if hasattr(X, "shape") else len(X.columns)
        self._coef = self.rng.random(n_features)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame | NDArray[Any]) -> NDArray[np.floating[Any]]:
        if not self._fitted or self._coef is None:
            raise ValueError("Model must be fitted before prediction")

        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return X_arr @ self._coef + self.rng.normal(0, 0.1, size=len(X_arr))


@pytest.fixture
def sample_multiindex_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample multi-series data with MultiIndex."""
    np.random.seed(42)
    n_series = 3
    n_samples_per_series = 100
    n_lags = 5

    series_ids = []
    dates = []
    lags_data = {f"lag_{i + 1}": [] for i in range(n_lags)}
    targets = []

    base_date = pd.Timestamp("2023-01-01")

    for series_id in [f"MT_{i:03d}" for i in range(1, n_series + 1)]:
        for t in range(n_samples_per_series):
            series_ids.append(series_id)
            dates.append(base_date + pd.Timedelta(hours=t))

            for lag in range(n_lags):
                lags_data[f"lag_{lag + 1}"].append(
                    np.random.randn() + (ord(series_id[-1]) - ord("0")) * 0.5
                )

            target = sum(lags_data[f"lag_{lag + 1}"][-1] * (0.5**lag) for lag in range(n_lags))
            targets.append(target + np.random.randn() * 0.1)

    index = pd.MultiIndex.from_arrays(
        [series_ids, dates],
        names=["level", "date"],
    )

    X = pd.DataFrame(lags_data, index=index)
    y = pd.Series(targets, index=index, name="target")

    return X, y


@pytest.fixture
def sample_flat_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample data without MultiIndex."""
    np.random.seed(42)
    n_samples = 300
    n_features = 5

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    X["series_id"] = np.repeat(["A", "B", "C"], n_samples // 3)

    y = X["feature_0"] * 0.5 + X["feature_1"] * 0.3 + np.random.randn(n_samples) * 0.1

    return X, pd.Series(y, name="target")


@pytest.fixture
def mock_model() -> MockModel:
    """Create a mock model for testing."""
    return MockModel(seed=42)


@pytest.fixture
def fitted_mock_model(
    sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
) -> MockModel:
    """Create a fitted mock model."""
    X, y = sample_multiindex_data
    model = MockModel(seed=42)
    model.fit(X, y.to_numpy())
    return model


@pytest.fixture
def fitted_rf_model(
    sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
) -> RandomForestRegressor:
    """Create a fitted RandomForest model."""
    X, y = sample_multiindex_data
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X.to_numpy(), y.to_numpy())
    return model


@pytest.fixture
def series_mapping() -> dict[str, str]:
    """Create a sample series-to-group mapping."""
    return {
        "MT_001": "group_A",
        "MT_002": "group_B",
        "MT_003": "group_A",
    }
