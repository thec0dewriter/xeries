"""Unit tests for newly added framework adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from timelens.adapters.darts import DartsAdapter
from timelens.adapters.sklearn import SklearnAdapter


class _DummyModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 1.5)


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "series": ["A", "A", "B", "B"],
            "lag_1": [1.0, 2.0, 3.0, 4.0],
            "lag_2": [0.5, 1.5, 2.5, 3.5],
        }
    )
    y = pd.Series([1.0, 1.2, 2.8, 3.9])
    return X, y


def test_sklearn_adapter_basic(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = sample_data
    adapter = SklearnAdapter(_DummyModel(), X, y, series_col="series")

    X_train, y_train = adapter.get_training_data()
    assert list(X_train.columns) == ["series", "lag_1", "lag_2"]
    assert len(y_train) == 4
    assert adapter.get_series_column() == "series"
    assert adapter.get_feature_names() == ["series", "lag_1", "lag_2"]
    assert np.allclose(adapter.predict(X_train), np.array([1.5, 1.5, 1.5, 1.5]))


def test_darts_adapter_basic(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = sample_data
    adapter = DartsAdapter(_DummyModel(), X, y, series_col="series")

    X_train, y_train = adapter.get_training_data()
    assert len(X_train) == 4
    assert len(y_train) == 4
    assert adapter.get_series_column() == "series"
    assert np.allclose(adapter.predict(X_train), np.array([1.5, 1.5, 1.5, 1.5]))


def test_adapter_rejects_missing_predict(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = sample_data

    class _NoPredict:
        pass

    with pytest.raises(ValueError, match="predict"):
        SklearnAdapter(_NoPredict(), X, y)

    with pytest.raises(ValueError, match="predict"):
        DartsAdapter(_NoPredict(), X, y)
