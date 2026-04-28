"""Unit tests for ``SkforecastAdapter`` that do not require ``skforecast``.

The integration suite under ``tests/integration/test_skforecast.py`` exercises
the adapter end-to-end against a real ``ForecasterRecursiveMultiSeries``. This
file holds focused unit tests where we can mock the forecaster, so that
regressions on adapter-level invariants are caught even on environments where
``skforecast`` is not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xeries.adapters.skforecast import SkforecastAdapter


class _RecordingEstimator:
    """Stub estimator that records the input to :meth:`predict`."""

    def __init__(self) -> None:
        self.last_X: Any = None

    def predict(self, X: Any) -> np.ndarray:
        self.last_X = X
        n = len(X) if hasattr(X, "__len__") else 1
        return np.zeros(n)


class _StubForecaster:
    """Minimal stand-in for ``ForecasterRecursiveMultiSeries``.

    Exposes the surface that
    :meth:`SkforecastAdapter._validate_forecaster` checks for
    (``estimator``, ``is_fitted``, ``create_train_X_y``) without importing
    ``skforecast``.
    """

    def __init__(self, estimator: _RecordingEstimator) -> None:
        self.estimator = estimator
        self.is_fitted = True

    def create_train_X_y(self, *args: Any, **kwargs: Any) -> tuple[pd.DataFrame, pd.Series]:
        raise NotImplementedError("Not exercised by predict-path tests.")


class TestSkforecastAdapterPredict:
    """Regression tests for #18: ``predict`` must preserve DataFrame column names.

    Sklearn / LightGBM emit ``"X does not have valid feature names, but ... was
    fitted with feature names"`` when a model fitted on a DataFrame is later
    called with a bare numpy array. Before #18, ``SkforecastAdapter.predict``
    eagerly converted any DataFrame input via ``X.to_numpy()`` and triggered
    that warning every time the adapter was used inside permutation /
    per-series importance loops.
    """

    def test_dataframe_passes_through_with_columns(self) -> None:
        """A DataFrame input reaches the estimator with column names intact."""
        estimator = _RecordingEstimator()
        adapter = SkforecastAdapter(_StubForecaster(estimator))

        X = pd.DataFrame(
            {
                "lag_1": [1.0, 2.0, 3.0],
                "lag_2": [4.0, 5.0, 6.0],
                "promo": [0, 1, 0],
            }
        )
        adapter.predict(X)

        assert isinstance(estimator.last_X, pd.DataFrame), (
            "Bug #18 regressed: SkforecastAdapter.predict() converted DataFrame "
            f"to {type(estimator.last_X).__name__}, stripping the feature names "
            "that LightGBM / sklearn rely on."
        )
        assert list(estimator.last_X.columns) == ["lag_1", "lag_2", "promo"]

    def test_numpy_array_passes_through_unchanged(self) -> None:
        """Non-DataFrame inputs (raw arrays) are still passed through untouched."""
        estimator = _RecordingEstimator()
        adapter = SkforecastAdapter(_StubForecaster(estimator))

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        adapter.predict(X)

        assert isinstance(estimator.last_X, np.ndarray)
        assert estimator.last_X.shape == (2, 2)

    def test_predict_returns_numpy_array(self) -> None:
        """``predict`` always wraps the underlying output in ``np.asarray``."""
        estimator = _RecordingEstimator()
        adapter = SkforecastAdapter(_StubForecaster(estimator))

        X = pd.DataFrame({"lag_1": [1.0, 2.0]})
        out = adapter.predict(X)

        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
