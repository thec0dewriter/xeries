"""Unit tests for adapter base class and contract enforcement."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from timelens.adapters.base import BaseAdapter


class IncompleteAdapter(BaseAdapter):
    """An adapter that doesn't implement all abstract methods."""

    def get_training_data(self, *args: Any, **kwargs: Any) -> tuple[pd.DataFrame, pd.Series]:
        return pd.DataFrame(), pd.Series(dtype=float)

    def predict(self, X: pd.DataFrame) -> NDArray[Any]:
        return np.array([])

    # Missing: get_feature_names and get_series_column


class CompleteAdapter(BaseAdapter):
    """A minimal complete adapter implementation for testing."""

    def __init__(self, X: pd.DataFrame, y: pd.Series, series_col: str = "series") -> None:
        self._X = X
        self._y = y
        self._series_col = series_col

    def get_training_data(self, *args: Any, **kwargs: Any) -> tuple[pd.DataFrame, pd.Series]:
        return self._X, self._y

    def predict(self, X: pd.DataFrame) -> NDArray[Any]:
        return np.zeros(len(X))

    def get_feature_names(self) -> list[str]:
        return list(self._X.columns)

    def get_series_column(self) -> str:
        return self._series_col


class TestBaseAdapterContract:
    """Test that BaseAdapter enforces its abstract method contract."""

    def test_cannot_instantiate_base_adapter(self) -> None:
        """BaseAdapter is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseAdapter()  # type: ignore[abstract]

    def test_incomplete_adapter_raises(self) -> None:
        """Adapter missing abstract methods cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAdapter()  # type: ignore[abstract]

    def test_complete_adapter_instantiates(self) -> None:
        """A fully implemented adapter can be instantiated."""
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0.1, 0.2])
        adapter = CompleteAdapter(X, y)
        assert adapter is not None


class TestCompleteAdapter:
    """Test the complete adapter implementation."""

    @pytest.fixture
    def adapter(self) -> CompleteAdapter:
        X = pd.DataFrame({
            "lag_1": [1.0, 2.0, 3.0],
            "lag_2": [0.5, 1.5, 2.5],
            "series": ["A", "A", "B"],
        })
        y = pd.Series([0.1, 0.2, 0.3], name="target")
        return CompleteAdapter(X, y, series_col="series")

    def test_get_training_data(self, adapter: CompleteAdapter) -> None:
        X, y = adapter.get_training_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 3
        assert len(y) == 3

    def test_predict(self, adapter: CompleteAdapter) -> None:
        X, _ = adapter.get_training_data()
        predictions = adapter.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_get_feature_names(self, adapter: CompleteAdapter) -> None:
        names = adapter.get_feature_names()
        assert isinstance(names, list)
        assert "lag_1" in names
        assert "lag_2" in names

    def test_get_series_column(self, adapter: CompleteAdapter) -> None:
        col = adapter.get_series_column()
        assert col == "series"
