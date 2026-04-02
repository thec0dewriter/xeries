"""Statistical significance helpers for explainability outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def bootstrap_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
) -> tuple[float, float]:
    """Estimate a bootstrap confidence interval for the mean."""
    if values.size == 0:
        raise ValueError("values must not be empty")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    rng = np.random.default_rng(random_state)
    means = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = float(np.mean(sample))

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1.0 - alpha))
    return lower, upper


def estimate_significance(
    importance_samples: dict[str, list[float] | np.ndarray],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Estimate confidence intervals and sign-consistency per feature."""
    rows: list[dict[str, Any]] = []

    for idx, (feature, raw_values) in enumerate(importance_samples.items()):
        values = np.asarray(raw_values, dtype=float)
        if values.size == 0:
            continue

        lower, upper = bootstrap_interval(
            values,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            random_state=None if random_state is None else random_state + idx,
        )
        mean_value = float(np.mean(values))
        significant = (lower > 0 and upper > 0) or (lower < 0 and upper < 0)

        rows.append({
            "feature": feature,
            "mean_importance": mean_value,
            "ci_lower": lower,
            "ci_upper": upper,
            "significant": significant,
            "n_samples": int(values.size),
        })

    if not rows:
        return pd.DataFrame(
            columns=["feature", "mean_importance", "ci_lower", "ci_upper", "significant", "n_samples"]
        )

    return pd.DataFrame(rows).sort_values("mean_importance", key=np.abs, ascending=False)
