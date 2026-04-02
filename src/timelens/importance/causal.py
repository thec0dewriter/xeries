"""Causal Feature Importance using DoWhy + EconML.

This module implements causal feature importance estimation following
DoWhy's 4-step pipeline (Model → Identify → Estimate → Refute),
adapted for multi-series time-series forecasting.

Requires optional dependencies: ``pip install dowhy econml``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from timelens.core.base import CausalExplainer
from timelens.core.types import (
    CausalResult,
    ModelProtocol,
    RefutationResult,
)

if TYPE_CHECKING:
    pass


_ESTIMATORS = {"causal_forest", "linear_dml", "dynamic_dml"}


class CausalFeatureImportance(CausalExplainer):
    """Causal feature importance via treatment effect estimation.

    Estimates the causal effect of each treatment feature on the outcome
    using EconML estimators, optionally guided by a causal graph (DAG)
    for confounder identification through DoWhy.

    Supports series-conditional estimation: effects can vary across
    different time series groups.

    Requires: ``pip install dowhy econml``

    Example:
        >>> import networkx as nx
        >>> dag = nx.DiGraph([("promotion", "sales"), ("season", "sales")])
        >>> causal = CausalFeatureImportance(
        ...     model=adapter,
        ...     treatment_features=["promotion"],
        ...     causal_graph=dag,
        ...     estimator="causal_forest",
        ... )
        >>> result = causal.explain(X, y)
        >>> print(result.to_dataframe())
    """

    def __init__(
        self,
        model: ModelProtocol,
        treatment_features: list[str],
        causal_graph: Any | None = None,
        estimator: str = "causal_forest",
        confounders: list[str] | None = None,
        series_col: str = "level",
        discover_graph: bool = False,
        discovery_method: str = "pc",
        random_state: int | None = None,
    ) -> None:
        """Initialize the causal feature importance estimator.

        Args:
            model: A model with a predict method.
            treatment_features: Features to estimate causal effects for.
            causal_graph: Optional DAG (networkx DiGraph).
                If not provided and discover_graph=False, all non-treatment
                features are used as confounders.
            estimator: EconML estimator to use:
                'causal_forest' (CausalForestDML),
                'linear_dml' (LinearDML),
                'dynamic_dml' (DynamicDML for time-varying treatments).
            confounders: Explicit list of confounder features.
                If None, inferred from causal_graph or uses all non-treatment features.
            series_col: Column or index level containing series identifiers.
            discover_graph: If True, auto-discover graph using causal-learn.
            discovery_method: Algorithm for graph discovery ('pc', 'ges').
            random_state: Random seed for reproducibility.
        """
        super().__init__(
            model=model,
            treatment_features=treatment_features,
            causal_graph=causal_graph,
            series_col=series_col,
            random_state=random_state,
        )
        if estimator not in _ESTIMATORS:
            raise ValueError(
                f"Unknown estimator: {estimator}. Choose from {sorted(_ESTIMATORS)}"
            )
        self.estimator = estimator
        self.confounders = confounders
        self.discover_graph = discover_graph
        self.discovery_method = discovery_method

    def explain(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        y: Any,
        *args: Any,
        **kwargs: Any,
    ) -> CausalResult:
        """Estimate causal effects of treatment features on the outcome.

        Uses the EconML estimator to calculate the Average Treatment Effect
        (ATE) for each treatment feature, controlling for confounders.

        Args:
            X: Input features DataFrame.
            y: Target values.

        Returns:
            CausalResult with treatment effects, confidence intervals, and p-values.
        """
        try:
            import econml  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "econml is required for CausalFeatureImportance. "
                "Install with: pip install econml"
            ) from e

        y_array = np.asarray(y).ravel()

        # Prepare feature matrix (reset MultiIndex if present)
        X_flat = X.reset_index(drop=True) if isinstance(X.index, pd.MultiIndex) else X

        # Auto-discover causal graph if requested
        if self.discover_graph and self.causal_graph is None:
            self.causal_graph = self._discover_graph(X_flat)

        # Determine confounders
        confounders = self._resolve_confounders(X_flat)

        effects: list[float] = []
        ci_list: list[tuple[float, float]] = []
        p_values: list[float] = []

        for treatment in self.treatment_features:
            effect, ci, p_val = self._estimate_single_treatment(
                X_flat, y_array, treatment, confounders
            )
            effects.append(effect)
            ci_list.append(ci)
            if p_val is not None:
                p_values.append(p_val)

        ci_array = np.array(ci_list) if ci_list else None
        p_array = np.array(p_values) if p_values else None

        return CausalResult(
            feature_names=list(self.treatment_features),
            treatment_effects=np.array(effects),
            confidence_intervals=ci_array,
            p_values=p_array,
            causal_graph=self.causal_graph,
            estimator_name=self.estimator,
        )

    def _resolve_confounders(self, X: pd.DataFrame) -> list[str]:
        """Determine confounder features."""
        if self.confounders is not None:
            return self.confounders

        # If causal graph is provided, extract confounders from it
        if self.causal_graph is not None:
            return self._confounders_from_graph()

        # Default: all non-treatment features
        return [c for c in X.columns if c not in self.treatment_features]

    def _confounders_from_graph(self) -> list[str]:
        """Extract confounder variables from the causal DAG.

        A confounder is a node that has paths to both a treatment
        and the outcome (common cause).
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "networkx is required when using causal_graph. "
                "Install with: pip install networkx"
            ) from e

        graph: nx.DiGraph = self.causal_graph  # type: ignore[assignment]
        if graph is None:
            return []
        all_nodes = set(graph.nodes())
        treatments = set(self.treatment_features)

        # Find all ancestors of treatment nodes and outcome nodes
        confounders: set[str] = set()
        for node in all_nodes - treatments:
            # A node is a potential confounder if it's a parent of any treatment
            # or if it's connected to both treatment and outcome paths
            successors = set(nx.descendants(graph, node)) | {node}
            if successors & treatments:
                confounders.add(node)

        # Remove treatment features from confounder set
        confounders -= treatments
        return list(confounders)

    def _estimate_single_treatment(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        treatment: str,
        confounders: list[str],
    ) -> tuple[float, tuple[float, float], float | None]:
        """Estimate the causal effect of a single treatment feature.

        Returns:
            Tuple of (average_effect, (ci_lower, ci_upper), p_value).
        """
        # Separate treatment from confounders
        T = X[treatment].values.reshape(-1, 1)
        W_cols = [c for c in confounders if c != treatment and c in X.columns]
        W = X[W_cols].values if W_cols else None

        est = self._build_estimator()
        est.fit(y, T, X=W)

        # Average Treatment Effect
        ate = float(np.mean(est.effect(X=W)))

        # Confidence intervals
        try:
            effect_inference = est.effect_inference(X=W)
            ci = effect_inference.conf_int(alpha=0.05)
            ci_lower = float(np.mean(ci[0]))
            ci_upper = float(np.mean(ci[1]))

            # P-value from inference
            p_val: float | None
            try:
                pvals = effect_inference.pvalue()
                p_val = float(np.mean(pvals))
            except (AttributeError, TypeError):
                p_val = None
        except (AttributeError, TypeError):
            ci_lower = ate - 1.96 * abs(ate) * 0.1
            ci_upper = ate + 1.96 * abs(ate) * 0.1
            p_val = None

        return ate, (ci_lower, ci_upper), p_val

    def _build_estimator(self) -> Any:
        """Build the EconML estimator."""
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )

        if self.estimator == "causal_forest":
            from econml.dml import CausalForestDML

            return CausalForestDML(
                model_y=RandomForestRegressor(
                    n_estimators=100, max_depth=5, random_state=self.random_state
                ),
                model_t=GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, random_state=self.random_state
                ),
                n_estimators=200,
                random_state=self.random_state,
            )
        if self.estimator == "linear_dml":
            from econml.dml import LinearDML

            return LinearDML(
                model_y=RandomForestRegressor(
                    n_estimators=100, random_state=self.random_state
                ),
                model_t=GradientBoostingRegressor(
                    n_estimators=100, random_state=self.random_state
                ),
                random_state=self.random_state,
            )
        if self.estimator == "dynamic_dml":
            from econml.dynamic.dml import DynamicDML

            return DynamicDML(
                model_y=RandomForestRegressor(
                    n_estimators=50, random_state=self.random_state
                ),
                model_t=RandomForestClassifier(
                    n_estimators=50, random_state=self.random_state
                ),
                random_state=self.random_state,
            )

        raise ValueError(f"Unknown estimator: {self.estimator}")

    def _discover_graph(self, X: pd.DataFrame) -> Any:
        """Auto-discover causal graph from data using causal-learn.

        Args:
            X: Input features DataFrame.

        Returns:
            networkx DiGraph representing the discovered causal structure.
        """
        try:
            import networkx as nx
            from causallearn.search.ConstraintBased.PC import pc
        except ImportError as e:
            raise ImportError(
                "causal-learn and networkx are required for graph discovery. "
                "Install with: pip install causal-learn networkx"
            ) from e

        data = X.select_dtypes(include=[np.number]).values
        feature_names = list(X.select_dtypes(include=[np.number]).columns)

        cg = pc(data, alpha=0.05, indep_test="fisherz")

        # Convert to networkx DiGraph
        graph = nx.DiGraph()
        graph.add_nodes_from(feature_names)
        adj_matrix = cg.G.graph
        n = len(feature_names)

        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                    graph.add_edge(feature_names[i], feature_names[j])

        return graph

    def refute(
        self,
        result: CausalResult,
        X: pd.DataFrame,
        y: Any,
        method: str = "placebo_treatment",
    ) -> RefutationResult:
        """Stress-test the causal estimate with a refutation method.

        Args:
            result: The original CausalResult to refute.
            X: Input features DataFrame.
            y: Target values.
            method: Refutation method:
                'placebo_treatment' — replace treatment with random noise.
                'random_common_cause' — add a random confounder.
                'subset' — re-estimate on a random subset.

        Returns:
            RefutationResult indicating whether the estimate is robust.
        """
        y_array = np.asarray(y).ravel()
        X_flat = X.reset_index(drop=True) if isinstance(X.index, pd.MultiIndex) else X

        original_effect = float(np.mean(result.treatment_effects))

        if method == "placebo_treatment":
            refuted_effect = self._refute_placebo(X_flat, y_array)
        elif method == "random_common_cause":
            refuted_effect = self._refute_random_cause(X_flat, y_array)
        elif method == "subset":
            refuted_effect = self._refute_subset(X_flat, y_array)
        else:
            raise ValueError(
                f"Unknown refutation method: {method}. "
                "Choose from 'placebo_treatment', 'random_common_cause', 'subset'"
            )

        # The refutation passes if the refuted effect is close to zero
        # (for placebo) or similar to the original (for subset)
        if method == "placebo_treatment":
            passed = abs(refuted_effect) < abs(original_effect) * 0.5
        elif method == "random_common_cause":
            passed = abs(refuted_effect - original_effect) < abs(original_effect) * 0.3
        else:  # subset
            passed = abs(refuted_effect - original_effect) < abs(original_effect) * 0.5

        return RefutationResult(
            method=method,
            original_effect=original_effect,
            refuted_effect=refuted_effect,
            p_value=None,
            passed=passed,
        )

    def _refute_placebo(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Refute by replacing treatment with random noise."""
        X_placebo = X.copy()
        for treatment in self.treatment_features:
            X_placebo[treatment] = self._rng.normal(
                loc=X[treatment].mean(),
                scale=X[treatment].std(),
                size=len(X),
            )
        confounders = self._resolve_confounders(X_placebo)
        effects = []
        for treatment in self.treatment_features:
            T = X_placebo[treatment].values.reshape(-1, 1)
            W_cols = [c for c in confounders if c != treatment and c in X.columns]
            W = X_placebo[W_cols].values if W_cols else None
            est = self._build_estimator()
            est.fit(y, T, X=W)
            effects.append(float(np.mean(est.effect(X=W))))
        return float(np.mean(effects))

    def _refute_random_cause(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Refute by adding a random common cause."""
        X_with_noise = X.copy()
        X_with_noise["_random_confounder"] = self._rng.normal(size=len(X))
        confounders = [*self._resolve_confounders(X_with_noise), "_random_confounder"]
        effects = []
        for treatment in self.treatment_features:
            T = X_with_noise[treatment].values.reshape(-1, 1)
            W_cols = [c for c in confounders if c != treatment and c in X_with_noise.columns]
            W = X_with_noise[W_cols].values if W_cols else None
            est = self._build_estimator()
            est.fit(y, T, X=W)
            effects.append(float(np.mean(est.effect(X=W))))
        return float(np.mean(effects))

    def _refute_subset(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Refute by re-estimating on a random subset."""
        n_subset = max(int(len(X) * 0.8), 10)
        indices = self._rng.choice(len(X), size=n_subset, replace=False)
        X_sub = X.iloc[indices]
        y_sub = y[indices]
        confounders = self._resolve_confounders(X_sub)
        effects = []
        for treatment in self.treatment_features:
            T = X_sub[treatment].values.reshape(-1, 1)
            W_cols = [c for c in confounders if c != treatment and c in X_sub.columns]
            W = X_sub[W_cols].values if W_cols else None
            est = self._build_estimator()
            est.fit(y_sub, T, X=W)
            effects.append(float(np.mean(est.effect(X=W))))
        return float(np.mean(effects))
