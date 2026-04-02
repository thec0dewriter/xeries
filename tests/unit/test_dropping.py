"""Unit tests for ConditionalDropImportance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from xeries.importance.dropping import ConditionalDropImportance

if TYPE_CHECKING:
    from tests.conftest import MockModel


class TestConditionalDropImportance:
    """Tests for ConditionalDropImportance."""

    def test_init(self, fitted_mock_model: MockModel) -> None:
        explainer = ConditionalDropImportance(
            model=fitted_mock_model,
            metric="mse",
            fill_strategy="mean",
        )
        assert explainer.fill_strategy == "mean"
        assert explainer.strategy == "auto"

    def test_invalid_fill_strategy(self, fitted_mock_model: MockModel) -> None:
        with pytest.raises(ValueError, match="Unknown fill_strategy"):
            ConditionalDropImportance(
                model=fitted_mock_model,
                fill_strategy="invalid",
            )

    def test_explain_mean(
        self,
        fitted_mock_model: MockModel,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = sample_multiindex_data
        explainer = ConditionalDropImportance(
            model=fitted_mock_model,
            metric="mse",
            fill_strategy="mean",
            random_state=42,
        )
        result = explainer.explain(X, y, features=["lag_1", "lag_2"])
        assert len(result.feature_names) == 2
        assert len(result.importances) == 2
        assert result.method == "conditional_drop"
        assert result.std is None
        assert result.n_repeats == 1

    def test_explain_median(
        self,
        fitted_mock_model: MockModel,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = sample_multiindex_data
        explainer = ConditionalDropImportance(
            model=fitted_mock_model,
            metric="mse",
            fill_strategy="median",
            random_state=42,
        )
        result = explainer.explain(X, y, features=["lag_1"])
        assert len(result.feature_names) == 1
        assert result.importances[0] >= 0  # dropping should increase error

    def test_explain_zero(
        self,
        fitted_mock_model: MockModel,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = sample_multiindex_data
        explainer = ConditionalDropImportance(
            model=fitted_mock_model,
            metric="mse",
            fill_strategy="zero",
            random_state=42,
        )
        result = explainer.explain(X, y, features=["lag_1"])
        assert len(result.importances) == 1

    def test_explain_noise(
        self,
        fitted_mock_model: MockModel,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = sample_multiindex_data
        explainer = ConditionalDropImportance(
            model=fitted_mock_model,
            metric="mse",
            fill_strategy="noise",
            random_state=42,
        )
        result = explainer.explain(X, y, features=["lag_1"])
        assert len(result.importances) == 1

    def test_explain_all_features(
        self,
        fitted_mock_model: MockModel,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = sample_multiindex_data
        explainer = ConditionalDropImportance(
            model=fitted_mock_model,
            metric="mse",
            fill_strategy="mean",
            random_state=42,
        )
        result = explainer.explain(X, y)
        assert len(result.feature_names) == len(X.columns)

    def test_to_dataframe(
        self,
        fitted_mock_model: MockModel,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = sample_multiindex_data
        explainer = ConditionalDropImportance(
            model=fitted_mock_model,
            metric="mse",
            fill_strategy="mean",
            random_state=42,
        )
        result = explainer.explain(X, y, features=["lag_1", "lag_2"])
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "importance" in df.columns

    def test_manual_strategy_with_groups(
        self,
        fitted_mock_model: MockModel,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = sample_multiindex_data
        groups = np.zeros(len(X), dtype=np.intp)
        groups[len(X) // 2 :] = 1

        explainer = ConditionalDropImportance(
            model=fitted_mock_model,
            metric="mse",
            fill_strategy="mean",
            strategy="manual",
            random_state=42,
        )
        result = explainer.explain(X, y, features=["lag_1"], groups=groups)
        assert len(result.importances) == 1

    def test_manual_strategy_no_groups_raises(
        self,
        fitted_mock_model: MockModel,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = sample_multiindex_data
        explainer = ConditionalDropImportance(
            model=fitted_mock_model,
            metric="mse",
            fill_strategy="mean",
            strategy="manual",
            random_state=42,
        )
        with pytest.raises(ValueError, match="strategy='manual'"):
            explainer.explain(X, y, features=["lag_1"])

    def test_different_metrics(
        self,
        fitted_mock_model: MockModel,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = sample_multiindex_data
        for metric in ("mse", "mae", "rmse"):
            explainer = ConditionalDropImportance(
                model=fitted_mock_model,
                metric=metric,
                fill_strategy="mean",
                random_state=42,
            )
            result = explainer.explain(X, y, features=["lag_1"])
            assert len(result.importances) == 1

    def test_baseline_score_in_result(
        self,
        fitted_mock_model: MockModel,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = sample_multiindex_data
        explainer = ConditionalDropImportance(
            model=fitted_mock_model,
            metric="mse",
            fill_strategy="mean",
            random_state=42,
        )
        result = explainer.explain(X, y, features=["lag_1"])
        assert result.baseline_score > 0
