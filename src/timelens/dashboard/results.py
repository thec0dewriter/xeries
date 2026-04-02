"""Dashboard result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from timelens.analysis.comparison import compare_rankings


@dataclass
class DashboardResult:
    """Container for outputs of all configured dashboard components."""

    interpretability: dict[str, Any] = field(default_factory=dict)
    error_analysis: dict[str, Any] = field(default_factory=dict)
    causal: dict[str, Any] = field(default_factory=dict)
    interactions: dict[str, Any] = field(default_factory=dict)

    def compare_rankings(self) -> pd.DataFrame:
        """Compare rankings across available feature-importance style methods."""
        candidates = {
            name: result
            for name, result in self.interpretability.items()
            if hasattr(result, "feature_names") and hasattr(result, "importances")
        }
        if len(candidates) < 2:
            return pd.DataFrame()
        return compare_rankings(candidates)

    def summary(self) -> pd.DataFrame:
        """Create a concise summary table of available dashboard outputs."""
        rows: list[dict[str, Any]] = []

        for name, result in self.interpretability.items():
            n_features = len(getattr(result, "feature_names", []))
            rows.append({"component": "interpretability", "name": name, "items": n_features})

        if self.error_analysis:
            rows.append({"component": "error_analysis", "name": "metrics", "items": len(self.error_analysis)})

        if self.causal:
            n_treatments = len(getattr(self.causal.get("treatment_effects"), "feature_names", []))
            rows.append({"component": "causal", "name": "treatment_effects", "items": n_treatments})

        if self.interactions:
            rows.append({"component": "interactions", "name": "shapiq", "items": 1})

        return pd.DataFrame(rows)
