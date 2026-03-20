"""Integration tests for skforecast adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.integration
class TestSkforecastIntegration:
    """Integration tests requiring skforecast."""

    @pytest.fixture
    def sample_series_data(self) -> pd.DataFrame:
        """Create sample multi-series data in skforecast format."""
        np.random.seed(42)
        n_periods = 100
        dates = pd.date_range("2023-01-01", periods=n_periods, freq="h")

        data = {}
        for series_id in ["MT_001", "MT_002", "MT_003"]:
            base = np.random.randn() * 10
            trend = np.linspace(0, 2, n_periods)
            noise = np.random.randn(n_periods) * 0.5
            data[series_id] = base + trend + noise

        return pd.DataFrame(data, index=dates)

    @pytest.mark.slow
    def test_adapter_with_forecaster(
        self,
        sample_series_data: pd.DataFrame,
    ) -> None:
        """Test adapter with skforecast ForecasterRecursiveMultiSeries (v0.21+)."""
        pytest.importorskip("skforecast")

        from sklearn.ensemble import RandomForestRegressor

        from skforecast.recursive import ForecasterRecursiveMultiSeries

        from tcpfi.adapters.skforecast import SkforecastAdapter

        forecaster = ForecasterRecursiveMultiSeries(
            estimator=RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42),
            lags=5,
        )
        forecaster.fit(series=sample_series_data)

        adapter = SkforecastAdapter(forecaster, series=sample_series_data)

        X, y = adapter.get_training_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        # skforecast 0.21+: wide/dict input uses DatetimeIndex + _level_skforecast column
        assert isinstance(X.index, pd.MultiIndex) or "_level_skforecast" in X.columns

        feature_names = adapter.get_feature_names()
        assert len(feature_names) == 5
        assert all(f.startswith("lag_") for f in feature_names)

        series_ids = adapter.get_series_ids()
        assert set(series_ids) == {"MT_001", "MT_002", "MT_003"}

        predictions = adapter.predict(X)
        assert len(predictions) == len(X)

    @pytest.mark.slow
    def test_full_pipeline_with_skforecast(
        self,
        sample_series_data: pd.DataFrame,
    ) -> None:
        """Test full pipeline: skforecast -> adapter -> explainer."""
        pytest.importorskip("skforecast")

        from sklearn.ensemble import RandomForestRegressor

        from skforecast.recursive import ForecasterRecursiveMultiSeries

        from tcpfi.adapters.skforecast import SkforecastAdapter
        from tcpfi.importance.permutation import ConditionalPermutationImportance

        forecaster = ForecasterRecursiveMultiSeries(
            estimator=RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42),
            lags=3,
        )
        forecaster.fit(series=sample_series_data)

        adapter = SkforecastAdapter(forecaster, series=sample_series_data)
        X, y = adapter.get_training_data()

        explainer = ConditionalPermutationImportance(
            model=adapter,
            metric="mse",
            strategy="auto",
            n_repeats=2,
            n_jobs=1,
            random_state=42,
        )

        result = explainer.compute(X, y, features=["lag_1", "lag_2"])

        assert len(result.feature_names) == 2
        assert len(result.importances) == 2
        df = result.to_dataframe()
        assert len(df) == 2

    @pytest.mark.slow
    def test_from_skforecast_helper(
        self,
        sample_series_data: pd.DataFrame,
    ) -> None:
        """Test from_skforecast helper function."""
        pytest.importorskip("skforecast")

        from sklearn.ensemble import RandomForestRegressor

        from skforecast.recursive import ForecasterRecursiveMultiSeries

        from tcpfi.adapters.skforecast import from_skforecast

        forecaster = ForecasterRecursiveMultiSeries(
            estimator=RandomForestRegressor(n_estimators=5, random_state=42),
            lags=3,
        )
        forecaster.fit(series=sample_series_data)

        adapter = from_skforecast(forecaster, series=sample_series_data)

        assert adapter.n_lags == 3
        assert adapter.get_series_column() in ("level", "_level_skforecast")

    @pytest.mark.slow
    def test_get_training_data_passes_series_at_call_time(
        self,
        sample_series_data: pd.DataFrame,
    ) -> None:
        """Series may be omitted from adapter if supplied only to get_training_data."""
        pytest.importorskip("skforecast")

        from sklearn.ensemble import RandomForestRegressor

        from skforecast.recursive import ForecasterRecursiveMultiSeries

        from tcpfi.adapters.skforecast import SkforecastAdapter

        forecaster = ForecasterRecursiveMultiSeries(
            estimator=RandomForestRegressor(n_estimators=5, random_state=42),
            lags=3,
        )
        forecaster.fit(series=sample_series_data)

        adapter = SkforecastAdapter(forecaster)
        X, y = adapter.get_training_data(series=sample_series_data)
        assert len(X) == len(y)
