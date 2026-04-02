"""Unit tests for CausalFeatureImportance and causal result types."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from xeries.core.types import CausalResult, RefutationResult
from xeries.importance.causal import CausalFeatureImportance


class TestCausalResult:
    """Tests for CausalResult dataclass."""

    def test_to_dataframe(self) -> None:
        result = CausalResult(
            feature_names=["promo", "price", "season"],
            treatment_effects=np.array([0.5, -0.3, 0.1]),
            confidence_intervals=np.array([[0.3, 0.7], [-0.5, -0.1], [-0.05, 0.25]]),
            p_values=np.array([0.01, 0.03, 0.15]),
            estimator_name="causal_forest",
        )
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "treatment_effect" in df.columns
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns
        assert "p_value" in df.columns
        assert len(df) == 3

    def test_to_dataframe_no_ci(self) -> None:
        result = CausalResult(
            feature_names=["a", "b"],
            treatment_effects=np.array([0.5, 0.3]),
        )
        df = result.to_dataframe()
        assert "ci_lower" not in df.columns

    def test_significant_features(self) -> None:
        result = CausalResult(
            feature_names=["a", "b", "c"],
            treatment_effects=np.array([0.5, -0.3, 0.1]),
            p_values=np.array([0.01, 0.03, 0.15]),
        )
        sig = result.significant_features(alpha=0.05)
        assert "a" in sig
        assert "b" in sig
        assert "c" not in sig

    def test_significant_features_no_pvalues(self) -> None:
        result = CausalResult(
            feature_names=["a", "b"],
            treatment_effects=np.array([0.5, 0.3]),
        )
        sig = result.significant_features()
        assert sig == ["a", "b"]


class TestRefutationResult:
    """Tests for RefutationResult dataclass."""

    def test_basic(self) -> None:
        r = RefutationResult(
            method="placebo_treatment",
            original_effect=0.5,
            refuted_effect=0.02,
            p_value=0.8,
            passed=True,
        )
        assert r.method == "placebo_treatment"
        assert r.passed is True


class TestCausalFeatureImportance:
    """Tests for CausalFeatureImportance."""

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame({
            "promo": rng.binomial(1, 0.3, n).astype(float),
            "price": rng.normal(10, 2, n),
            "season": rng.choice(4, n).astype(float),
            "lag_1": rng.normal(0, 1, n),
            "lag_2": rng.normal(0, 1, n),
        })
        y = 2.0 * X["promo"] - 0.5 * X["price"] + rng.normal(0, 0.5, n)
        return X, y.values

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        model = MagicMock()
        model.predict = MagicMock(
            side_effect=lambda x: np.zeros(len(x))
        )
        return model

    def test_init(self, mock_model: MagicMock) -> None:
        explainer = CausalFeatureImportance(
            model=mock_model,
            treatment_features=["promo", "price"],
            estimator="causal_forest",
        )
        assert explainer.treatment_features == ["promo", "price"]
        assert explainer.estimator == "causal_forest"

    def test_invalid_estimator(self, mock_model: MagicMock) -> None:
        with pytest.raises(ValueError, match="Unknown estimator"):
            CausalFeatureImportance(
                model=mock_model,
                treatment_features=["promo"],
                estimator="invalid",
            )

    def test_resolve_confounders_explicit(self, mock_model: MagicMock) -> None:
        explainer = CausalFeatureImportance(
            model=mock_model,
            treatment_features=["promo"],
            confounders=["price", "season"],
        )
        X = pd.DataFrame({"promo": [1], "price": [10], "season": [1], "lag_1": [0]})
        confounders = explainer._resolve_confounders(X)
        assert confounders == ["price", "season"]

    def test_resolve_confounders_default(self, mock_model: MagicMock) -> None:
        explainer = CausalFeatureImportance(
            model=mock_model,
            treatment_features=["promo"],
        )
        X = pd.DataFrame({"promo": [1], "price": [10], "season": [1]})
        confounders = explainer._resolve_confounders(X)
        assert "promo" not in confounders
        assert "price" in confounders
        assert "season" in confounders

    def test_explain_with_mock_econml(
        self,
        mock_model: MagicMock,
        sample_data: tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        X, y = sample_data

        # Create a mock estimator
        mock_est = MagicMock()
        mock_est.effect.return_value = np.array([0.5] * len(X))
        mock_inference = MagicMock()
        mock_inference.conf_int.return_value = (
            np.array([0.3] * len(X)),
            np.array([0.7] * len(X)),
        )
        mock_inference.pvalue.return_value = np.array([0.01] * len(X))
        mock_est.effect_inference.return_value = mock_inference

        explainer = CausalFeatureImportance(
            model=mock_model,
            treatment_features=["promo"],
            estimator="causal_forest",
            random_state=42,
        )

        mock_econml = MagicMock()
        with patch.dict("sys.modules", {"econml": mock_econml}), patch.object(
            explainer, "_build_estimator", return_value=mock_est
        ):
            result = explainer.explain(X, y)

        assert isinstance(result, CausalResult)
        assert len(result.feature_names) == 1
        assert result.feature_names[0] == "promo"
        assert result.estimator_name == "causal_forest"
        assert len(result.treatment_effects) == 1

    def test_explain_multiple_treatments(
        self,
        mock_model: MagicMock,
        sample_data: tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        X, y = sample_data

        mock_est = MagicMock()
        mock_est.effect.return_value = np.array([0.3] * len(X))
        mock_est.effect_inference.side_effect = AttributeError("no inference")

        explainer = CausalFeatureImportance(
            model=mock_model,
            treatment_features=["promo", "price"],
            estimator="causal_forest",
            random_state=42,
        )

        mock_econml = MagicMock()
        with patch.dict("sys.modules", {"econml": mock_econml}), patch.object(
            explainer, "_build_estimator", return_value=mock_est
        ):
            result = explainer.explain(X, y)

        assert len(result.feature_names) == 2
        assert len(result.treatment_effects) == 2

    def test_explain_with_multiindex(
        self,
        mock_model: MagicMock,
    ) -> None:
        rng = np.random.default_rng(42)
        n = 50
        idx = pd.MultiIndex.from_arrays(
            [["A"] * 25 + ["B"] * 25, range(n)],
            names=["level", "time"],
        )
        X = pd.DataFrame(
            {"promo": rng.binomial(1, 0.3, n).astype(float), "lag_1": rng.normal(0, 1, n)},
            index=idx,
        )
        y = rng.normal(0, 1, n)

        mock_est = MagicMock()
        mock_est.effect.return_value = np.array([0.5] * n)
        mock_est.effect_inference.side_effect = AttributeError()

        explainer = CausalFeatureImportance(
            model=mock_model,
            treatment_features=["promo"],
            random_state=42,
        )

        mock_econml = MagicMock()
        with patch.dict("sys.modules", {"econml": mock_econml}), patch.object(
            explainer, "_build_estimator", return_value=mock_est
        ):
            result = explainer.explain(X, y)

        assert isinstance(result, CausalResult)

    def test_import_error_without_econml(
        self,
        mock_model: MagicMock,
        sample_data: tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        X, y = sample_data
        explainer = CausalFeatureImportance(
            model=mock_model,
            treatment_features=["promo"],
        )

        with patch.dict("sys.modules", {"econml": None}), pytest.raises(
            ImportError, match="econml is required"
        ):
            explainer.explain(X, y)

    def test_refute_placebo(
        self,
        mock_model: MagicMock,
        sample_data: tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        X, y = sample_data

        original_result = CausalResult(
            feature_names=["promo"],
            treatment_effects=np.array([0.5]),
        )

        mock_est = MagicMock()
        mock_est.effect.return_value = np.array([0.02] * len(X))
        mock_est.effect_inference.side_effect = AttributeError()

        explainer = CausalFeatureImportance(
            model=mock_model,
            treatment_features=["promo"],
            random_state=42,
        )

        with patch.object(explainer, "_build_estimator", return_value=mock_est):
            refutation = explainer.refute(original_result, X, y, method="placebo_treatment")

        assert isinstance(refutation, RefutationResult)
        assert refutation.method == "placebo_treatment"
        assert refutation.original_effect == 0.5

    def test_refute_subset(
        self,
        mock_model: MagicMock,
        sample_data: tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        X, y = sample_data

        original_result = CausalResult(
            feature_names=["promo"],
            treatment_effects=np.array([0.5]),
        )

        mock_est = MagicMock()
        mock_est.effect.return_value = np.array([0.48] * 160)
        mock_est.effect_inference.side_effect = AttributeError()

        explainer = CausalFeatureImportance(
            model=mock_model,
            treatment_features=["promo"],
            random_state=42,
        )

        with patch.object(explainer, "_build_estimator", return_value=mock_est):
            refutation = explainer.refute(original_result, X, y, method="subset")

        assert refutation.method == "subset"

    def test_refute_invalid_method(
        self,
        mock_model: MagicMock,
        sample_data: tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        X, y = sample_data
        result = CausalResult(
            feature_names=["promo"],
            treatment_effects=np.array([0.5]),
        )
        explainer = CausalFeatureImportance(
            model=mock_model,
            treatment_features=["promo"],
        )
        with pytest.raises(ValueError, match="Unknown refutation method"):
            explainer.refute(result, X, y, method="invalid")

    def test_confounders_from_graph(self, mock_model: MagicMock) -> None:
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")

        dag = nx.DiGraph([
            ("season", "promo"),
            ("season", "sales"),
            ("promo", "sales"),
            ("price", "sales"),
        ])

        explainer = CausalFeatureImportance(
            model=mock_model,
            treatment_features=["promo", "price"],
            causal_graph=dag,
        )

        confounders = explainer._confounders_from_graph()
        assert "season" in confounders
        assert "promo" not in confounders
        assert "price" not in confounders
