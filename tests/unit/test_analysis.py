"""Unit tests for analysis utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from timelens.analysis.comparison import compare_rankings
from timelens.analysis.error import ErrorAnalyzer
from timelens.analysis.significance import bootstrap_interval, estimate_significance
from timelens.core.types import FeatureImportanceResult


def test_error_analyzer_outputs() -> None:
    analyzer = ErrorAnalyzer(metrics=["mse", "mae", "rmse"])
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 2.0, 5.0])
    series = np.array(["A", "A", "B", "B"])

    global_metrics = analyzer.compute(y_true, y_pred)
    assert set(global_metrics.keys()) == {"mse", "mae", "rmse"}

    by_series = analyzer.by_series(y_true, y_pred, series)
    assert set(by_series.columns) >= {"series", "mse", "mae", "rmse"}
    assert len(by_series) == 2

    by_window = analyzer.by_time_window(y_true, y_pred, window_size=2)
    assert len(by_window) == 2
    assert "window_start" in by_window.columns


def test_compare_rankings() -> None:
    r1 = FeatureImportanceResult(
        feature_names=["a", "b", "c"],
        importances=np.array([0.6, 0.3, 0.1]),
    )
    r2 = FeatureImportanceResult(
        feature_names=["a", "b", "c"],
        importances=np.array([0.5, 0.2, 0.1]),
    )

    matrix = compare_rankings({"pfi": r1, "drop": r2})
    assert matrix.shape == (2, 2)
    assert matrix.loc["pfi", "pfi"] == 1.0


def test_significance_helpers() -> None:
    values = np.array([0.1, 0.15, 0.2, 0.18, 0.11])
    lo, hi = bootstrap_interval(values, n_bootstrap=100, random_state=42)
    assert lo <= hi

    summary = estimate_significance({"lag_1": values}, n_bootstrap=100, random_state=42)
    assert "feature" in summary.columns
    assert summary.iloc[0]["feature"] == "lag_1"
