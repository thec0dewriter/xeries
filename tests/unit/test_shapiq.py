"""Unit tests for ConditionalSHAPIQ and SHAPIQResult."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from timelens.importance.shapiq import ConditionalSHAPIQ, SHAPIQResult


def _make_mock_interaction_values(
    n_features: int, max_order: int = 2
) -> MagicMock:
    """Create a mock shapiq InteractionValues object."""
    iv = MagicMock()

    # Mock first-order values: 1D array of shape (n_features,)
    first_order = np.random.default_rng(42).standard_normal(n_features)
    # Mock second-order values: 2D array of shape (n_features, n_features)
    second_order = np.random.default_rng(42).standard_normal((n_features, n_features))

    def get_n_order(n: int) -> np.ndarray:
        if n == 1:
            return first_order
        if n == 2:
            return second_order
        raise ValueError(f"Order {n} not mocked")

    iv.get_n_order_values = MagicMock(side_effect=get_n_order)
    return iv


class TestSHAPIQResult:
    """Tests for SHAPIQResult container."""

    @pytest.fixture
    def result(self) -> SHAPIQResult:
        feature_names = ["lag_1", "lag_2", "lag_3"]
        ivs = [_make_mock_interaction_values(3) for _ in range(5)]
        return SHAPIQResult(
            interaction_values=ivs,
            feature_names=feature_names,
            max_order=2,
            index="k-SII",
        )

    def test_init(self, result: SHAPIQResult) -> None:
        assert result.n_instances == 5
        assert result.max_order == 2
        assert result.index == "k-SII"
        assert len(result.feature_names) == 3

    def test_get_first_order(self, result: SHAPIQResult) -> None:
        df = result.get_first_order()
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "mean_abs_attribution" in df.columns
        assert len(df) == 3
        # Should be sorted descending
        assert df["mean_abs_attribution"].is_monotonic_decreasing

    def test_get_second_order(self, result: SHAPIQResult) -> None:
        df = result.get_second_order()
        assert isinstance(df, pd.DataFrame)
        assert "feature_1" in df.columns
        assert "feature_2" in df.columns
        assert "mean_abs_interaction" in df.columns
        # For 3 features, we should have C(3,2) = 3 pairs
        assert len(df) == 3

    def test_get_second_order_insufficient_order(self) -> None:
        result = SHAPIQResult(
            interaction_values=[],
            feature_names=["a", "b"],
            max_order=1,
            index="SV",
        )
        with pytest.raises(ValueError, match="max_order must be >= 2"):
            result.get_second_order()

    def test_to_dataframe_order1(self, result: SHAPIQResult) -> None:
        df = result.to_dataframe(order=1)
        assert "feature" in df.columns
        assert "mean_abs_attribution" in df.columns

    def test_to_dataframe_order2(self, result: SHAPIQResult) -> None:
        df = result.to_dataframe(order=2)
        assert "feature_1" in df.columns
        assert "mean_abs_interaction" in df.columns

    def test_to_dataframe_invalid_order(self, result: SHAPIQResult) -> None:
        with pytest.raises(ValueError, match="to_dataframe supports order 1 or 2"):
            result.to_dataframe(order=3)


class TestConditionalSHAPIQ:
    """Tests for ConditionalSHAPIQ explainer."""

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create sample multi-series data with MultiIndex."""
        rng = np.random.default_rng(42)
        n_per_series = 20
        feature_names = ["lag_1", "lag_2", "lag_3"]

        dfs = []
        for sid in ["A", "B"]:
            data = rng.standard_normal((n_per_series, len(feature_names)))
            idx = pd.MultiIndex.from_arrays(
                [[sid] * n_per_series, range(n_per_series)],
                names=["level", "time"],
            )
            df = pd.DataFrame(data, index=idx, columns=feature_names)
            dfs.append(df)

        X = pd.concat(dfs)
        y = pd.Series(rng.standard_normal(len(X)), index=X.index, name="target")
        return X, y

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        model = MagicMock()
        model.predict = MagicMock(
            side_effect=lambda x: np.zeros(len(x)) if hasattr(x, "__len__") else np.array([0.0])
        )
        return model

    def test_init(
        self, mock_model: MagicMock, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, _ = sample_data
        explainer = ConditionalSHAPIQ(
            model=mock_model,
            background_data=X,
            series_col="level",
            max_order=2,
        )
        assert explainer.max_order == 2
        assert explainer.index == "k-SII"
        assert len(explainer._series_backgrounds) == 2

    def test_series_backgrounds_prepared(
        self, mock_model: MagicMock, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, _ = sample_data
        explainer = ConditionalSHAPIQ(
            model=mock_model,
            background_data=X,
            series_col="level",
        )
        assert "A" in explainer._series_backgrounds
        assert "B" in explainer._series_backgrounds

    def test_explain(
        self,
        mock_model: MagicMock,
        sample_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, _ = sample_data
        n_features = len(X.columns)

        # Create a mock shapiq module
        mock_shapiq = MagicMock()
        mock_explainer_instance = MagicMock()
        mock_iv = _make_mock_interaction_values(n_features, max_order=2)
        mock_explainer_instance.explain.return_value = mock_iv
        mock_shapiq.TabularExplainer.return_value = mock_explainer_instance

        explainer = ConditionalSHAPIQ(
            model=mock_model,
            background_data=X,
            series_col="level",
            max_order=2,
            random_state=42,
        )

        with patch.dict("sys.modules", {"shapiq": mock_shapiq}):
            result = explainer.explain(X.iloc[:2])

        assert isinstance(result, SHAPIQResult)
        assert result.n_instances == 2
        assert result.max_order == 2
        assert result.feature_names == list(X.columns)

    def test_explain_with_feature_names(
        self,
        mock_model: MagicMock,
        sample_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, _ = sample_data
        selected_features = ["lag_1", "lag_2"]

        mock_shapiq = MagicMock()
        mock_explainer_instance = MagicMock()
        mock_iv = _make_mock_interaction_values(len(selected_features))
        mock_explainer_instance.explain.return_value = mock_iv
        mock_shapiq.TabularExplainer.return_value = mock_explainer_instance

        explainer = ConditionalSHAPIQ(
            model=mock_model,
            background_data=X,
            series_col="level",
            max_order=2,
            random_state=42,
        )

        with patch.dict("sys.modules", {"shapiq": mock_shapiq}):
            result = explainer.explain(X.iloc[:1], feature_names=selected_features)
        assert result.feature_names == selected_features

    def test_import_error_without_shapiq(
        self,
        mock_model: MagicMock,
        sample_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, _ = sample_data
        explainer = ConditionalSHAPIQ(
            model=mock_model,
            background_data=X,
            series_col="level",
        )

        with patch.dict("sys.modules", {"shapiq": None}), pytest.raises(
            ImportError, match="shapiq package is required"
        ):
            explainer.explain(X.iloc[:1])
