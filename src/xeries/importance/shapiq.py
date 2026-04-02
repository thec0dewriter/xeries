"""Conditional SHAP-IQ implementation for feature interaction analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from xeries.core.base import AttributionExplainer
from xeries.core.types import BaseResult, ModelProtocol

if TYPE_CHECKING:
    pass


class SHAPIQResult(BaseResult):
    """Container for SHAP-IQ interaction results.

    Stores interaction values from shapiq alongside metadata for
    downstream analysis and visualization.

    Attributes:
        interaction_values: Raw shapiq InteractionValues objects per instance.
        feature_names: List of feature names.
        max_order: Maximum interaction order computed.
        index: The interaction index used (e.g. 'k-SII').
        n_instances: Number of instances explained.
    """

    def __init__(
        self,
        interaction_values: list[Any],
        feature_names: list[str],
        max_order: int,
        index: str,
    ) -> None:
        self.interaction_values = interaction_values
        self.feature_names = feature_names
        self.max_order = max_order
        self.index = index
        self.n_instances = len(interaction_values)

    def get_first_order(self) -> pd.DataFrame:
        """Get mean absolute first-order (main effect) attributions.

        Returns:
            DataFrame with features and their mean absolute attribution.
        """
        all_first = []
        for iv in self.interaction_values:
            first_order = iv.get_n_order_values(1)
            all_first.append(first_order)

        stacked = np.array(all_first)
        mean_abs = np.abs(stacked).mean(axis=0)

        return pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_attribution": mean_abs,
        }).sort_values("mean_abs_attribution", ascending=False)

    def get_second_order(self) -> pd.DataFrame:
        """Get mean absolute second-order (pairwise) interactions.

        Returns:
            DataFrame with feature pairs and their mean absolute interaction.

        Raises:
            ValueError: If max_order < 2.
        """
        if self.max_order < 2:
            raise ValueError("max_order must be >= 2 to retrieve second-order interactions")

        rows: list[dict[str, Any]] = []
        n_features = len(self.feature_names)

        all_second = []
        for iv in self.interaction_values:
            second_order = iv.get_n_order_values(2)
            all_second.append(second_order)

        stacked = np.array(all_second)
        mean_abs = np.abs(stacked).mean(axis=0)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                rows.append({
                    "feature_1": self.feature_names[i],
                    "feature_2": self.feature_names[j],
                    "mean_abs_interaction": mean_abs[i, j],
                })

        return pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False)

    def to_dataframe(self, order: int = 1) -> pd.DataFrame:
        """Convert results to a DataFrame for the specified interaction order.

        Args:
            order: Interaction order (1 = main effects, 2 = pairwise, etc.).

        Returns:
            DataFrame with interaction values.
        """
        if order == 1:
            return self.get_first_order()
        if order == 2:
            return self.get_second_order()
        raise ValueError(f"to_dataframe supports order 1 or 2, got {order}")


class ConditionalSHAPIQ(AttributionExplainer):
    """Conditional SHAP-IQ explainer for feature interaction analysis.

    Computes Shapley interaction indices using series-specific background
    data. This extends the ConditionalSHAP approach to capture not just
    individual feature attributions but also interaction effects between
    features.

    Requires the ``shapiq`` package (install with ``pip install shapiq``).

    Example:
        >>> explainer = ConditionalSHAPIQ(model, X, series_col='level', max_order=2)
        >>> result = explainer.explain(X.iloc[:5])
        >>> print(result.get_second_order())
    """

    def __init__(
        self,
        model: ModelProtocol,
        background_data: pd.DataFrame,
        series_col: str = "level",
        max_order: int = 2,
        index: str = "k-SII",
        n_background_samples: int = 100,
        budget: int = 2048,
        random_state: int | None = None,
    ) -> None:
        """Initialize the conditional SHAP-IQ explainer.

        Args:
            model: A model with a predict method.
            background_data: Full dataset to sample background from.
            series_col: Column or index level containing series identifiers.
            max_order: Maximum interaction order to compute (2 = pairwise).
            index: Interaction index type ('k-SII', 'SII', 'STII', 'FSII', 'SV').
            n_background_samples: Number of background samples per series.
            budget: Computation budget for the shapiq approximator.
            random_state: Random seed for reproducibility.
        """
        super().__init__(model, background_data, random_state)
        self.series_col = series_col
        self.max_order = max_order
        self.index = index
        self.n_background_samples = n_background_samples
        self.budget = budget

        self._series_backgrounds: dict[Any, pd.DataFrame] = {}
        self._prepare_series_backgrounds()

    def _prepare_series_backgrounds(self) -> None:
        """Prepare background datasets for each series."""
        series_ids = self._get_series_ids(self.background_data)
        unique_series = series_ids.unique()

        for series_id in unique_series:
            mask = series_ids == series_id
            series_data = self.background_data[mask]

            if len(series_data) > self.n_background_samples:
                indices = self._rng.choice(
                    len(series_data),
                    size=self.n_background_samples,
                    replace=False,
                )
                series_data = series_data.iloc[indices]

            self._series_backgrounds[series_id] = series_data

    def _get_series_ids(self, X: pd.DataFrame) -> pd.Series:
        """Extract series identifiers from DataFrame."""
        if isinstance(X.index, pd.MultiIndex) and self.series_col in X.index.names:
            return X.index.get_level_values(self.series_col).to_series(index=X.index)

        if self.series_col in X.columns:
            return X[self.series_col]

        if self.series_col == "level" and "_level_skforecast" in X.columns:
            return X["_level_skforecast"]

        raise KeyError(
            f"Series column '{self.series_col}' not found in DataFrame columns or index"
        )

    def explain(
        self,
        X: pd.DataFrame,
        feature_names: list[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> SHAPIQResult:
        """Compute SHAP-IQ interaction values for the given instances.

        Uses series-specific background data for conditional computation.

        Args:
            X: Instances to explain.
            feature_names: Names of features. If None, uses X.columns.

        Returns:
            SHAPIQResult containing interaction values for all instances.
        """
        try:
            import shapiq
        except ImportError as e:
            raise ImportError(
                "shapiq package is required for ConditionalSHAPIQ. "
                "Install it with: pip install shapiq"
            ) from e

        feature_names = feature_names or list(X.columns)
        series_ids = self._get_series_ids(X)

        all_interaction_values: list[Any] = []

        for idx, (_i, row) in enumerate(X.iterrows()):
            series_id = series_ids.iloc[idx]
            background = self._get_background_for_series(series_id)

            explainer = shapiq.TabularExplainer(
                model=self.model.predict,
                data=background[feature_names].values,
                index=self.index,
                max_order=self.max_order,
                random_state=self.random_state,
            )

            row_values = row[feature_names].values.reshape(1, -1)
            interaction_values = explainer.explain(row_values[0], budget=self.budget)
            all_interaction_values.append(interaction_values)

        return SHAPIQResult(
            interaction_values=all_interaction_values,
            feature_names=feature_names,
            max_order=self.max_order,
            index=self.index,
        )

    def _get_background_for_series(self, series_id: Any) -> pd.DataFrame:
        """Get background data for a specific series."""
        if series_id not in self._series_backgrounds:
            all_backgrounds = pd.concat(list(self._series_backgrounds.values()))
            indices = self._rng.choice(
                len(all_backgrounds),
                size=min(self.n_background_samples, len(all_backgrounds)),
                replace=False,
            )
            return all_backgrounds.iloc[indices]

        return self._series_backgrounds[series_id]
