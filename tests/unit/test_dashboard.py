"""Unit tests for dashboard orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xeries.dashboard.core import Dashboard


class _DummyModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Keep deterministic behavior for tests.
        return (X["lag_1"].to_numpy() * 0.1 + X["lag_2"].to_numpy() * 0.2)


def _make_data() -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame(
        {
            "level": ["A", "A", "A", "B", "B", "B"],
            "lag_1": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "lag_2": [0.5, 0.7, 0.9, 1.2, 1.4, 1.6],
        }
    )
    y = np.array([0.2, 0.35, 0.5, 0.35, 0.5, 0.65])
    return X, y


def test_dashboard_compute_interpretability_and_error() -> None:
    X, y = _make_data()
    dashboard = Dashboard(model=_DummyModel(), X=X, y=y, series_col="level")

    results = (
        dashboard.add_interpretability(methods=["permutation", "dropping"], features=["lag_1", "lag_2"])
        .add_error_analysis(metrics=["mse", "mae"], by_series=True, by_time_window=True, window_size=2)
        .compute()
    )

    assert "permutation" in results.interpretability
    assert "dropping" in results.interpretability
    assert "global_metrics" in results.error_analysis
    assert "by_series" in results.error_analysis
    assert "by_time_window" in results.error_analysis


def test_dashboard_report_generation(tmp_path: str) -> None:
    # Only run this test when Jinja2 is available.
    import pytest

    pytest.importorskip("jinja2")

    X, y = _make_data()
    dashboard = Dashboard(model=_DummyModel(), X=X, y=y, series_col="level")
    dashboard.add_interpretability(methods=["permutation"], features=["lag_1", "lag_2"]).compute()

    output = dashboard.generate_report(path=f"{tmp_path}/report.html")
    assert output.exists()
