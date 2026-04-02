"""Dashboard orchestration entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from timelens.dashboard.components.causal import CausalComponent
from timelens.dashboard.components.error_analysis import ErrorAnalysisComponent
from timelens.dashboard.components.interactions import InteractionComponent
from timelens.dashboard.components.interpretability import InterpretabilityComponent
from timelens.dashboard.report import show_html_report, write_html_report
from timelens.dashboard.results import DashboardResult


@dataclass
class Dashboard:
    """Builder-style API that orchestrates explainability components."""

    model: Any
    X: pd.DataFrame
    y: Any
    series_col: str = "level"

    def __post_init__(self) -> None:
        self._interpretability_component: InterpretabilityComponent | None = None
        self._error_component: ErrorAnalysisComponent | None = None
        self._causal_component: CausalComponent | None = None
        self._interaction_component: InteractionComponent | None = None
        self._results: DashboardResult | None = None

    def add_interpretability(
        self,
        methods: list[str],
        strategy: str = "auto",
        features: list[str] | None = None,
        n_repeats: int = 5,
        random_state: int | None = None,
    ) -> Dashboard:
        self._interpretability_component = InterpretabilityComponent(
            model=self.model,
            methods=methods,
            strategy=strategy,
            features=features,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        return self

    def add_error_analysis(
        self,
        metrics: list[str] | None = None,
        by_series: bool = True,
        by_time_window: bool = False,
        window_size: int = 24,
    ) -> Dashboard:
        self._error_component = ErrorAnalysisComponent(
            model=self.model,
            metrics=metrics or ["mse", "mae", "rmse"],
            by_series=by_series,
            by_time_window=by_time_window,
            window_size=window_size,
        )
        return self

    def add_causal_analysis(
        self,
        treatment_features: list[str],
        estimator: str = "causal_forest",
        causal_graph: Any | None = None,
        random_state: int | None = None,
    ) -> Dashboard:
        self._causal_component = CausalComponent(
            model=self.model,
            treatment_features=treatment_features,
            estimator=estimator,
            causal_graph=causal_graph,
            random_state=random_state,
        )
        return self

    def add_interactions(
        self,
        method: str = "shapiq",
        max_order: int = 2,
        n_samples: int | None = None,
        random_state: int | None = None,
    ) -> Dashboard:
        self._interaction_component = InteractionComponent(
            model=self.model,
            method=method,
            max_order=max_order,
            n_samples=n_samples,
            random_state=random_state,
        )
        return self

    def compute(self) -> DashboardResult:
        """Run all configured components and return combined results."""
        result = DashboardResult()

        if self._interpretability_component is not None:
            result.interpretability = self._interpretability_component.compute(self.X, self.y)

        if self._error_component is not None:
            result.error_analysis = self._error_component.compute(self.X, self.y, self.series_col)

        if self._causal_component is not None:
            result.causal = {
                "treatment_effects": self._causal_component.compute(self.X, self.y),
            }

        if self._interaction_component is not None:
            result.interactions = self._interaction_component.compute(self.X)

        self._results = result
        return result

    def plot_all(self) -> dict[str, Any]:
        """Return plot handles for available interpretability outputs."""
        if self._results is None:
            raise RuntimeError("Call compute() before plot_all()")

        try:
            from timelens.visualization import plot_importance_bar
        except ImportError as e:
            raise ImportError("matplotlib is required for plotting") from e

        plots: dict[str, Any] = {}
        for name, result in self._results.interpretability.items():
            if hasattr(result, "importances"):
                plots[name] = plot_importance_bar(result)
        return plots

    def generate_report(self, path: str | Path, title: str = "TimeLens Dashboard Report") -> Path:
        """Write an HTML report to disk."""
        if self._results is None:
            raise RuntimeError("Call compute() before generate_report()")
        return write_html_report(path=path, results=self._results, title=title)

    def generate_scorecard(self, path: str | Path) -> Path:
        """Generate a lightweight scorecard (HTML for now)."""
        return self.generate_report(path=path, title="TimeLens Scorecard")

    def show(self, title: str = "TimeLens Dashboard Report") -> Path:
        """Open an interactive HTML report in the browser."""
        if self._results is None:
            raise RuntimeError("Call compute() before show()")
        return show_html_report(self._results, title=title)
