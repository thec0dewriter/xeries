"""Interpretability dashboard component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from timelens.importance.dropping import ConditionalDropImportance
from timelens.importance.permutation import ConditionalPermutationImportance
from timelens.importance.shap import ConditionalSHAP


@dataclass
class InterpretabilityComponent:
    """Run selected interpretability explainers."""

    model: Any
    methods: list[str] = field(default_factory=lambda: ["permutation"])
    strategy: str = "auto"
    features: list[str] | None = None
    n_repeats: int = 5
    random_state: int | None = None

    def compute(self, X: pd.DataFrame, y: Any) -> dict[str, Any]:
        outputs: dict[str, Any] = {}

        if "permutation" in self.methods:
            pfi = ConditionalPermutationImportance(
                model=self.model,
                strategy=self.strategy,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
            )
            outputs["permutation"] = pfi.explain(X, y, features=self.features)

        if "dropping" in self.methods:
            dropping = ConditionalDropImportance(
                model=self.model,
                strategy=self.strategy,
                random_state=self.random_state,
            )
            outputs["dropping"] = dropping.explain(X, y, features=self.features)

        if "shap" in self.methods:
            shap_explainer = ConditionalSHAP(
                model=self.model,
                background_data=X,
                random_state=self.random_state,
            )
            outputs["shap"] = shap_explainer.explain(X[self.features] if self.features else X)

        return outputs
