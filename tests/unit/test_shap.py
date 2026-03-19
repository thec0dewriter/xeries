"""Unit tests for ConditionalSHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from tcpfi.core.types import SHAPResult


class TestConditionalSHAP:
    """Tests for ConditionalSHAP."""

    @pytest.fixture
    def small_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create small dataset for SHAP testing (SHAP is slow)."""
        np.random.seed(42)
        n_samples = 30

        index = pd.MultiIndex.from_arrays(
            [
                np.repeat(["A", "B", "C"], n_samples // 3),
                pd.date_range("2023-01-01", periods=n_samples, freq="h"),
            ],
            names=["level", "date"],
        )

        X = pd.DataFrame(
            {
                "lag_1": np.random.randn(n_samples),
                "lag_2": np.random.randn(n_samples),
            },
            index=index,
        )
        y = pd.Series(
            X["lag_1"] * 0.5 + X["lag_2"] * 0.3 + np.random.randn(n_samples) * 0.1,
            index=index,
        )

        return X, y

    @pytest.fixture
    def fitted_model_small(
        self,
        small_data: tuple[pd.DataFrame, pd.Series],
    ) -> RandomForestRegressor:
        """Create fitted model for small data."""
        X, y = small_data
        model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X.to_numpy(), y.to_numpy())
        return model

    @pytest.mark.slow
    def test_init(
        self,
        small_data: tuple[pd.DataFrame, pd.Series],
        fitted_model_small: RandomForestRegressor,
    ) -> None:
        """Test explainer initialization."""
        from tcpfi.importance.shap import ConditionalSHAP

        X, _ = small_data
        explainer = ConditionalSHAP(
            predict_fn=fitted_model_small.predict,
            background_data=X,
            series_col="level",
            n_background_samples=10,
            random_state=42,
        )

        assert explainer.series_col == "level"
        assert explainer.n_background_samples == 10
        assert len(explainer._series_backgrounds) == 3

    @pytest.mark.slow
    def test_explain(
        self,
        small_data: tuple[pd.DataFrame, pd.Series],
        fitted_model_small: RandomForestRegressor,
    ) -> None:
        """Test computing SHAP values."""
        from tcpfi.importance.shap import ConditionalSHAP

        X, _ = small_data
        explainer = ConditionalSHAP(
            predict_fn=fitted_model_small.predict,
            background_data=X,
            series_col="level",
            n_background_samples=5,
            random_state=42,
        )

        result = explainer.explain(X.iloc[:3])

        assert isinstance(result, SHAPResult)
        assert result.shap_values.shape == (3, 2)
        assert len(result.base_values) == 3
        assert result.feature_names == ["lag_1", "lag_2"]

    @pytest.mark.slow
    def test_explain_instance(
        self,
        small_data: tuple[pd.DataFrame, pd.Series],
        fitted_model_small: RandomForestRegressor,
    ) -> None:
        """Test explaining a single instance."""
        from tcpfi.importance.shap import ConditionalSHAP

        X, _ = small_data
        explainer = ConditionalSHAP(
            predict_fn=fitted_model_small.predict,
            background_data=X,
            series_col="level",
            n_background_samples=5,
            random_state=42,
        )

        result = explainer.explain_instance(X.iloc[0])

        assert result.shap_values.shape == (1, 2)

    @pytest.mark.slow
    def test_mean_abs_shap(
        self,
        small_data: tuple[pd.DataFrame, pd.Series],
        fitted_model_small: RandomForestRegressor,
    ) -> None:
        """Test computing mean absolute SHAP values."""
        from tcpfi.importance.shap import ConditionalSHAP

        X, _ = small_data
        explainer = ConditionalSHAP(
            predict_fn=fitted_model_small.predict,
            background_data=X,
            series_col="level",
            n_background_samples=5,
            random_state=42,
        )

        result = explainer.explain(X.iloc[:5])
        mean_abs = result.mean_abs_shap()

        assert isinstance(mean_abs, pd.DataFrame)
        assert "feature" in mean_abs.columns
        assert "mean_abs_shap" in mean_abs.columns
        assert len(mean_abs) == 2

    @pytest.mark.slow
    def test_global_importance(
        self,
        small_data: tuple[pd.DataFrame, pd.Series],
        fitted_model_small: RandomForestRegressor,
    ) -> None:
        """Test computing global importance."""
        from tcpfi.importance.shap import ConditionalSHAP

        X, _ = small_data
        explainer = ConditionalSHAP(
            predict_fn=fitted_model_small.predict,
            background_data=X,
            series_col="level",
            n_background_samples=5,
            random_state=42,
        )

        global_imp = explainer.global_importance(X, n_samples=5)

        assert isinstance(global_imp, pd.DataFrame)
        assert len(global_imp) == 2

    def test_series_backgrounds_prepared(
        self,
        small_data: tuple[pd.DataFrame, pd.Series],
        fitted_model_small: RandomForestRegressor,
    ) -> None:
        """Test that series-specific backgrounds are prepared."""
        from tcpfi.importance.shap import ConditionalSHAP

        X, _ = small_data
        explainer = ConditionalSHAP(
            predict_fn=fitted_model_small.predict,
            background_data=X,
            series_col="level",
            n_background_samples=5,
            random_state=42,
        )

        assert "A" in explainer._series_backgrounds
        assert "B" in explainer._series_backgrounds
        assert "C" in explainer._series_backgrounds
        assert len(explainer._series_backgrounds["A"]) <= 5

    @pytest.mark.slow
    def test_shap_values_shape_consistency(
        self,
        small_data: tuple[pd.DataFrame, pd.Series],
        fitted_model_small: RandomForestRegressor,
    ) -> None:
        """Test that SHAP values have correct shape for varying sample sizes.

        This test verifies the fix for the issue where shap_values[0] was
        incorrectly extracting single elements instead of feature arrays.
        """
        from tcpfi.importance.shap import ConditionalSHAP

        X, _ = small_data
        n_features = X.shape[1]
        explainer = ConditionalSHAP(
            predict_fn=fitted_model_small.predict,
            background_data=X,
            series_col="level",
            n_background_samples=5,
            random_state=42,
        )

        # Test single instance
        result_single = explainer.explain(X.iloc[[0]])
        assert result_single.shap_values.shape == (1, n_features)
        assert result_single.shap_values.ndim == 2

        # Test multiple instances
        result_multi = explainer.explain(X.iloc[:5])
        assert result_multi.shap_values.shape == (5, n_features)
        assert result_multi.shap_values.ndim == 2

        # Verify mean_abs_shap works correctly (requires 2D array)
        mean_abs = result_multi.mean_abs_shap()
        assert len(mean_abs) == n_features
        assert all(mean_abs["mean_abs_shap"] >= 0)
