"""Error analysis dashboard component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from timelens.analysis.error import ErrorAnalyzer


@dataclass
class ErrorAnalysisComponent:
    """Compute global, per-series and per-window error metrics."""

    model: Any
    metrics: list[str] = field(default_factory=lambda: ["mse", "mae", "rmse"])
    by_series: bool = True
    by_time_window: bool = False
    window_size: int = 24

    def compute(self, X: pd.DataFrame, y: Any, series_col: str) -> dict[str, Any]:
        analyzer = ErrorAnalyzer(metrics=self.metrics)
        y_pred = self.model.predict(X)

        output: dict[str, Any] = {"global_metrics": analyzer.compute(y, y_pred)}

        if self.by_series:
            if series_col in X.columns:
                series_ids = X[series_col]
            elif isinstance(X.index, pd.MultiIndex) and series_col in X.index.names:
                series_ids = X.index.get_level_values(series_col)
            else:
                raise KeyError(f"Series column '{series_col}' not found in X")

            output["by_series"] = analyzer.by_series(y, y_pred, series_ids)

        if self.by_time_window:
            output["by_time_window"] = analyzer.by_time_window(
                y,
                y_pred,
                window_size=self.window_size,
            )

        return output
