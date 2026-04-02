"""Interaction dashboard component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from xeries.importance.shapiq import ConditionalSHAPIQ


@dataclass
class InteractionComponent:
    """Run SHAP-IQ interaction analysis in dashboard workflows."""

    model: Any
    method: str = "shapiq"
    max_order: int = 2
    n_samples: int | None = None
    random_state: int | None = None

    def compute(self, X: pd.DataFrame) -> dict[str, Any]:
        if self.method != "shapiq":
            raise ValueError("Only method='shapiq' is currently supported")

        X_input = X
        if self.n_samples is not None and self.n_samples < len(X):
            X_input = X.sample(n=self.n_samples, random_state=self.random_state)

        explainer = ConditionalSHAPIQ(
            model=self.model,
            background_data=X,
            max_order=self.max_order,
            random_state=self.random_state,
        )
        return {"shapiq": explainer.explain(X_input)}
