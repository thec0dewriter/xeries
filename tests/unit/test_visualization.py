"""Unit tests for visualization functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from xeries.core.types import FeatureImportanceResult, SHAPResult

if TYPE_CHECKING:
    pass


@pytest.fixture
def sample_importance_result() -> FeatureImportanceResult:
    """Create a sample FeatureImportanceResult for testing."""
    return FeatureImportanceResult(
        feature_names=["lag_1", "lag_2", "lag_3", "day_of_week"],
        importances=np.array([0.15, 0.08, 0.03, 0.12]),
        std=np.array([0.02, 0.01, 0.005, 0.015]),
        baseline_score=0.5,
        permuted_scores={
            "lag_1": [0.65, 0.63, 0.67],
            "lag_2": [0.58, 0.57, 0.59],
            "lag_3": [0.53, 0.52, 0.54],
            "day_of_week": [0.62, 0.61, 0.63],
        },
        method="conditional_permutation",
        n_repeats=3,
    )


@pytest.fixture
def sample_importance_result_no_std() -> FeatureImportanceResult:
    """Create a FeatureImportanceResult without std for testing."""
    return FeatureImportanceResult(
        feature_names=["f1", "f2"],
        importances=np.array([0.1, 0.2]),
        std=None,
        baseline_score=0.5,
    )


@pytest.fixture
def sample_shap_result() -> SHAPResult:
    """Create a sample SHAPResult for testing."""
    rng = np.random.default_rng(42)
    n_samples = 10
    n_features = 4
    feature_names = ["lag_1", "lag_2", "lag_3", "day_of_week"]

    return SHAPResult(
        shap_values=rng.standard_normal((n_samples, n_features)),
        base_values=rng.standard_normal(n_samples),
        feature_names=feature_names,
        data=rng.standard_normal((n_samples, n_features)),
    )


@pytest.fixture
def multiple_importance_results(
    sample_importance_result: FeatureImportanceResult,
) -> dict[str, FeatureImportanceResult]:
    """Create multiple FeatureImportanceResults for heatmap testing."""
    result_b = FeatureImportanceResult(
        feature_names=sample_importance_result.feature_names,
        importances=np.array([0.10, 0.12, 0.05, 0.08]),
        std=np.array([0.01, 0.02, 0.01, 0.01]),
        baseline_score=0.45,
        method="conditional_permutation",
        n_repeats=3,
    )
    return {
        "condition_A": sample_importance_result,
        "condition_B": result_b,
    }


class TestPlotImportanceBar:
    """Tests for plot_importance_bar."""

    def test_returns_figure_and_axes(
        self, sample_importance_result: FeatureImportanceResult
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_importance_bar

        fig, ax = plot_importance_bar(sample_importance_result)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_max_features_limits_bars(
        self, sample_importance_result: FeatureImportanceResult
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_importance_bar

        fig, ax = plot_importance_bar(sample_importance_result, max_features=2)
        # Should only show top 2 features
        assert len(ax.patches) == 2
        plt.close(fig)

    def test_custom_title(
        self, sample_importance_result: FeatureImportanceResult
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_importance_bar

        fig, ax = plot_importance_bar(sample_importance_result, title="Custom Title")
        assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_no_std_bars(
        self, sample_importance_result_no_std: FeatureImportanceResult
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_importance_bar

        fig, ax = plot_importance_bar(sample_importance_result_no_std, show_std=True)
        # Should not error even though std is None
        assert fig is not None
        plt.close(fig)

    def test_show_std_false(
        self, sample_importance_result: FeatureImportanceResult
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_importance_bar

        fig, ax = plot_importance_bar(sample_importance_result, show_std=False)
        # No error bars when show_std=False
        assert fig is not None
        plt.close(fig)

    def test_custom_axes(
        self, sample_importance_result: FeatureImportanceResult
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_importance_bar

        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_importance_bar(sample_importance_result, ax=ax_ext)
        assert ax is ax_ext
        plt.close(fig_ext)

    def test_no_max_features(
        self, sample_importance_result: FeatureImportanceResult
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_importance_bar

        fig, ax = plot_importance_bar(sample_importance_result, max_features=None)
        assert len(ax.patches) == 4
        plt.close(fig)


class TestPlotImportanceHeatmap:
    """Tests for plot_importance_heatmap."""

    def test_returns_figure_and_axes(
        self, multiple_importance_results: dict[str, FeatureImportanceResult]
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_importance_heatmap

        fig, ax = plot_importance_heatmap(multiple_importance_results)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_custom_title(
        self, multiple_importance_results: dict[str, FeatureImportanceResult]
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_importance_heatmap

        fig, ax = plot_importance_heatmap(
            multiple_importance_results, title="Heatmap Title"
        )
        assert ax.get_title() == "Heatmap Title"
        plt.close(fig)

    def test_feature_filter(
        self, multiple_importance_results: dict[str, FeatureImportanceResult]
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_importance_heatmap

        fig, ax = plot_importance_heatmap(
            multiple_importance_results, features=["lag_1", "lag_2"]
        )
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert len(ytick_labels) == 2
        plt.close(fig)

    def test_no_annotations(
        self, multiple_importance_results: dict[str, FeatureImportanceResult]
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_importance_heatmap

        fig, ax = plot_importance_heatmap(
            multiple_importance_results, annot=False
        )
        # Should work without annotations
        assert fig is not None
        plt.close(fig)


class TestPlotShapBar:
    """Tests for plot_shap_bar."""

    def test_returns_figure_and_axes(self, sample_shap_result: SHAPResult) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_shap_bar

        fig, ax = plot_shap_bar(sample_shap_result)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_max_features(self, sample_shap_result: SHAPResult) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_shap_bar

        fig, ax = plot_shap_bar(sample_shap_result, max_features=2)
        assert len(ax.patches) == 2
        plt.close(fig)

    def test_custom_title(self, sample_shap_result: SHAPResult) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_shap_bar

        fig, ax = plot_shap_bar(sample_shap_result, title="SHAP Title")
        assert ax.get_title() == "SHAP Title"
        plt.close(fig)


class TestPlotImportanceComparison:
    """Tests for plot_importance_comparison."""

    def test_returns_figure_and_axes(
        self,
        multiple_importance_results: dict[str, FeatureImportanceResult],
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xeries.visualization.plots import plot_importance_comparison

        fig, ax = plot_importance_comparison(multiple_importance_results, top_n=3)
        assert fig is not None
        assert ax is not None
        plt.close(fig)
