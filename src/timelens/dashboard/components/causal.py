"""Causal dashboard component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from timelens.importance.causal import CausalFeatureImportance


@dataclass
class CausalComponent:
    """Run causal feature importance as part of dashboard execution."""

    model: Any
    treatment_features: list[str]
    estimator: str = "causal_forest"
    causal_graph: Any | None = None
    random_state: int | None = None

    def compute(self, X: pd.DataFrame, y: Any) -> Any:
        explainer = CausalFeatureImportance(
            model=self.model,
            treatment_features=self.treatment_features,
            estimator=self.estimator,
            causal_graph=self.causal_graph,
            random_state=self.random_state,
        )
        return explainer.explain(X, y)
