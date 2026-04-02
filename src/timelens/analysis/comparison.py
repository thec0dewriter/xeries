"""Comparison utilities for explainability method outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timelens.core.types import FeatureImportanceResult


def compare_rankings(results: dict[str, FeatureImportanceResult]) -> pd.DataFrame:
    """Compare feature rankings across methods using Kendall-style agreement.

    Returns a symmetric matrix with pairwise rank correlations.
    """
    methods = list(results.keys())
    matrix = np.eye(len(methods), dtype=float)

    for i, left in enumerate(methods):
        for j, right in enumerate(methods):
            if i >= j:
                continue
            tau = _kendall_tau_from_results(results[left], results[right])
            matrix[i, j] = tau
            matrix[j, i] = tau

    return pd.DataFrame(matrix, index=methods, columns=methods)


def _kendall_tau_from_results(a: FeatureImportanceResult, b: FeatureImportanceResult) -> float:
    shared = [f for f in a.feature_names if f in b.feature_names]
    if len(shared) < 2:
        return 0.0

    rank_a = _rank_map(a, shared)
    rank_b = _rank_map(b, shared)

    concordant = 0
    discordant = 0
    n = len(shared)

    for i in range(n):
        for j in range(i + 1, n):
            fi = shared[i]
            fj = shared[j]
            da = rank_a[fi] - rank_a[fj]
            db = rank_b[fi] - rank_b[fj]
            sign = da * db
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1

    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 0.0
    return float((concordant - discordant) / total_pairs)


def _rank_map(result: FeatureImportanceResult, features: list[str]) -> dict[str, Any]:
    idx = {f: i for i, f in enumerate(result.feature_names)}
    values = {f: float(result.importances[idx[f]]) for f in features}
    sorted_features = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    return {f: rank for rank, (f, _v) in enumerate(sorted_features, start=1)}
