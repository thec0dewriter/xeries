"""Error analysis utilities for dashboard and standalone workflows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


MetricFn = Callable[[np.ndarray, np.ndarray], float]


class ErrorAnalyzer:
    """Compute global, per-series, and time-window error metrics."""

    def __init__(self, metrics: list[str] | None = None) -> None:
        self.metrics = metrics or ["mse", "mae", "rmse"]
        self._metric_map: dict[str, MetricFn] = {
            "mse": self._mse,
            "mae": self._mae,
            "rmse": self._rmse,
        }

        unknown = [m for m in self.metrics if m not in self._metric_map]
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}. Choose from {sorted(self._metric_map)}")

    def compute(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> dict[str, float]:
        """Compute global metrics."""
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        return {m: self._metric_map[m](yt, yp) for m in self.metrics}

    def by_series(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        series_ids: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        """Compute metrics per series."""
        df = pd.DataFrame({
            "y_true": np.asarray(y_true, dtype=float).reshape(-1),
            "y_pred": np.asarray(y_pred, dtype=float).reshape(-1),
            "series": np.asarray(series_ids),
        })

        rows: list[dict[str, Any]] = []
        for series_value, group in df.groupby("series", sort=True):
            row: dict[str, Any] = {"series": series_value, "n_samples": len(group)}
            yt = group["y_true"].to_numpy()
            yp = group["y_pred"].to_numpy()
            for metric in self.metrics:
                row[metric] = self._metric_map[metric](yt, yp)
            rows.append(row)

        return pd.DataFrame(rows).sort_values("series").reset_index(drop=True)

    def by_time_window(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        window_size: int,
    ) -> pd.DataFrame:
        """Compute metrics over fixed-size rolling windows (non-overlapping)."""
        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_pred, dtype=float).reshape(-1)

        if len(yt) != len(yp):
            raise ValueError("y_true and y_pred must have equal length")

        rows: list[dict[str, Any]] = []
        for start in range(0, len(yt), window_size):
            end = min(start + window_size, len(yt))
            w_true = yt[start:end]
            w_pred = yp[start:end]
            row: dict[str, Any] = {
                "window_start": start,
                "window_end": end - 1,
                "n_samples": len(w_true),
            }
            for metric in self.metrics:
                row[metric] = self._metric_map[metric](w_true, w_pred)
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
