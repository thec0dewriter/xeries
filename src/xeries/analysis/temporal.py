"""Temporal importance analysis over fixed windows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xeries.core.types import FeatureImportanceResult


class TemporalImportance:
    """Compute feature importance over sequential windows."""

    def __init__(self, explainer: Any, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.explainer = explainer
        self.window_size = window_size

    def compute(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run the wrapped explainer over non-overlapping windows."""
        y_array = np.asarray(y)
        rows: list[dict[str, Any]] = []

        for start in range(0, len(X), self.window_size):
            end = min(start + self.window_size, len(X))
            X_w = X.iloc[start:end]
            y_w = y_array[start:end]

            result = self.explainer.explain(X_w, y_w, features=features)
            if not isinstance(result, FeatureImportanceResult):
                raise TypeError("TemporalImportance expects FeatureImportanceResult outputs")

            for feature, importance in zip(result.feature_names, result.importances, strict=True):
                rows.append({
                    "window_start": start,
                    "window_end": end - 1,
                    "feature": feature,
                    "importance": float(importance),
                })

        return pd.DataFrame(rows)
