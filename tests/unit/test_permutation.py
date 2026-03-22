"""Unit tests for ConditionalPermutationImportance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from tcpfi.core.types import FeatureImportanceResult
from tcpfi.importance.permutation import ConditionalPermutationImportance
from tcpfi.partitioners.manual import ManualPartitioner


class TestConditionalPermutationImportance:
    """Tests for ConditionalPermutationImportance."""

    def test_init(
        self,
        fitted_rf_model: RandomForestRegressor,
    ) -> None:
        """Test explainer initialization."""
        explainer = ConditionalPermutationImportance(
            model=fitted_rf_model,
            metric="mse",
            strategy="auto",
            n_repeats=3,
        )
        assert explainer.strategy == "auto"
        assert explainer.n_repeats == 3

    def test_compute_auto_strategy(
        self,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
        fitted_rf_model: RandomForestRegressor,
    ) -> None:
        """Test computing importance with auto (tree-based) strategy."""
        X, y = sample_multiindex_data
        explainer = ConditionalPermutationImportance(
            model=fitted_rf_model,
            metric="mse",
            strategy="auto",
            n_repeats=2,
            n_jobs=1,
            random_state=42,
        )

        result = explainer.compute(X, y, features=["lag_1", "lag_2"])

        assert isinstance(result, FeatureImportanceResult)
        assert len(result.feature_names) == 2
        assert len(result.importances) == 2
        assert result.std is not None
        assert len(result.std) == 2
        assert result.method == "conditional_permutation"

    def test_compute_manual_strategy(
        self,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
        fitted_rf_model: RandomForestRegressor,
        series_mapping: dict[str, str],
    ) -> None:
        """Test computing importance with manual strategy."""
        X, y = sample_multiindex_data
        partitioner = ManualPartitioner(series_mapping, series_col="level")

        explainer = ConditionalPermutationImportance(
            model=fitted_rf_model,
            metric="mse",
            strategy="manual",
            partitioner=partitioner,
            n_repeats=2,
            n_jobs=1,
            random_state=42,
        )

        result = explainer.compute(X, y, features=["lag_1"])

        assert isinstance(result, FeatureImportanceResult)
        assert len(result.feature_names) == 1

    def test_compute_with_groups(
        self,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
        fitted_rf_model: RandomForestRegressor,
    ) -> None:
        """Test computing importance with pre-defined groups."""
        X, y = sample_multiindex_data
        groups = np.zeros(len(X), dtype=np.intp)
        groups[len(X) // 2 :] = 1

        explainer = ConditionalPermutationImportance(
            model=fitted_rf_model,
            metric="mse",
            strategy="manual",
            n_repeats=2,
            n_jobs=1,
            random_state=42,
        )

        result = explainer.compute(X, y, features=["lag_1"], groups=groups)

        assert isinstance(result, FeatureImportanceResult)

    def test_compute_all_features(
        self,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
        fitted_rf_model: RandomForestRegressor,
    ) -> None:
        """Test computing importance for all features when none specified."""
        X, y = sample_multiindex_data
        explainer = ConditionalPermutationImportance(
            model=fitted_rf_model,
            metric="mse",
            n_repeats=1,
            n_jobs=1,
            random_state=42,
        )

        result = explainer.compute(X, y)

        assert len(result.feature_names) == len(X.columns)

    def test_to_dataframe(
        self,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
        fitted_rf_model: RandomForestRegressor,
    ) -> None:
        """Test converting results to DataFrame."""
        X, y = sample_multiindex_data
        explainer = ConditionalPermutationImportance(
            model=fitted_rf_model,
            metric="mse",
            n_repeats=2,
            n_jobs=1,
            random_state=42,
        )

        result = explainer.compute(X, y, features=["lag_1", "lag_2"])
        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "importance" in df.columns
        assert "std" in df.columns
        assert len(df) == 2

    def test_different_metrics(
        self,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
        fitted_rf_model: RandomForestRegressor,
    ) -> None:
        """Test using different metrics."""
        X, y = sample_multiindex_data

        for metric in ["mse", "mae", "rmse"]:
            explainer = ConditionalPermutationImportance(
                model=fitted_rf_model,
                metric=metric,
                n_repeats=1,
                n_jobs=1,
                random_state=42,
            )
            result = explainer.compute(X, y, features=["lag_1"])
            assert len(result.importances) == 1

    def test_custom_metric(
        self,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
        fitted_rf_model: RandomForestRegressor,
    ) -> None:
        """Test using a custom metric function."""
        X, y = sample_multiindex_data

        def custom_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            """Custom metric function matching MetricFunction protocol."""
            return float(np.mean((y_true - y_pred) ** 2))

        explainer = ConditionalPermutationImportance(
            model=fitted_rf_model,
            metric=custom_metric,
            n_repeats=1,
            n_jobs=1,
            random_state=42,
        )

        result = explainer.compute(X, y, features=["lag_1"])
        assert len(result.importances) == 1

    def test_invalid_metric(
        self,
        fitted_rf_model: RandomForestRegressor,
    ) -> None:
        """Test error with invalid metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            ConditionalPermutationImportance(
                model=fitted_rf_model,
                metric="invalid_metric",
            )

    def test_manual_strategy_no_groups(
        self,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
        fitted_rf_model: RandomForestRegressor,
    ) -> None:
        """Test error when manual strategy used without groups."""
        X, y = sample_multiindex_data
        explainer = ConditionalPermutationImportance(
            model=fitted_rf_model,
            metric="mse",
            strategy="manual",
            n_repeats=1,
            n_jobs=1,
        )

        with pytest.raises(ValueError, match="provide 'groups' or a 'partitioner'"):
            explainer.compute(X, y, features=["lag_1"])

    def test_baseline_score_in_result(
        self,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
        fitted_rf_model: RandomForestRegressor,
    ) -> None:
        """Test that baseline score is included in result."""
        X, y = sample_multiindex_data
        explainer = ConditionalPermutationImportance(
            model=fitted_rf_model,
            metric="mse",
            n_repeats=1,
            n_jobs=1,
            random_state=42,
        )

        result = explainer.compute(X, y, features=["lag_1"])

        assert result.baseline_score > 0
        assert "lag_1" in result.permuted_scores

    def test_reproducibility(
        self,
        sample_multiindex_data: tuple[pd.DataFrame, pd.Series],
        fitted_rf_model: RandomForestRegressor,
    ) -> None:
        """Test that results are reproducible with same random state."""
        X, y = sample_multiindex_data

        explainer1 = ConditionalPermutationImportance(
            model=fitted_rf_model,
            metric="mse",
            n_repeats=2,
            n_jobs=1,
            random_state=42,
        )
        result1 = explainer1.compute(X, y, features=["lag_1"])

        explainer2 = ConditionalPermutationImportance(
            model=fitted_rf_model,
            metric="mse",
            n_repeats=2,
            n_jobs=1,
            random_state=42,
        )
        result2 = explainer2.compute(X, y, features=["lag_1"])

        np.testing.assert_array_almost_equal(result1.importances, result2.importances)
