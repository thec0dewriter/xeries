"""Conditional SHAP implementation for time series models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from timelens.core.base import AttributionExplainer
from timelens.core.types import ModelProtocol, SHAPResult

if TYPE_CHECKING:
    pass


class ConditionalSHAP(AttributionExplainer):
    """Conditional SHAP explainer for multi-series forecasting models.

    This explainer computes SHAP values using series-specific background
    data, ensuring that the baseline/expected value is computed from
    data belonging to the same series as the instance being explained.

    Example:
        >>> explainer = ConditionalSHAP(model, X, series_col='level')
        >>> result = explainer.explain(X.iloc[:10])
        >>> print(result.mean_abs_shap())
    """

    def __init__(
        self,
        model: ModelProtocol,
        background_data: pd.DataFrame,
        series_col: str = "level",
        n_background_samples: int = 100,
        random_state: int | None = None,
    ) -> None:
        """Initialize the conditional SHAP explainer.

        Args:
            model: A model with a predict method.
            background_data: Full dataset to sample background from.
            series_col: Column or index level containing series identifiers.
            n_background_samples: Number of background samples per series.
            random_state: Random seed for reproducibility.
        """
        super().__init__(model, background_data, random_state)
        self.series_col = series_col
        self.n_background_samples = n_background_samples

        self._rng = np.random.default_rng(random_state)
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

        # skforecast 0.21+ uses ordinal codes in _level_skforecast when level is not present
        if self.series_col == "level" and "_level_skforecast" in X.columns:
            return X["_level_skforecast"]

        raise KeyError(f"Series column '{self.series_col}' not found in DataFrame columns or index")

    def explain(
        self,
        X: pd.DataFrame,
        feature_names: list[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> SHAPResult:  # type: ignore[override]
        """Compute SHAP values for the given instances.

        Uses KernelSHAP with series-specific background data.

        Args:
            X: Instances to explain.
            feature_names: Names of features. If None, uses X.columns.

        Returns:
            SHAPResult containing SHAP values and base values.
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "shap package is required for ConditionalSHAP. Install it with: pip install shap"
            ) from e

        feature_names = feature_names or list(X.columns)
        series_ids = self._get_series_ids(X)

        all_shap_values = []
        all_base_values = []

        for idx, (_i, row) in enumerate(X.iterrows()):
            series_id = series_ids.iloc[idx]
            background = self._get_background_for_series(series_id)

            row_df = pd.DataFrame([row], columns=X.columns)
            explainer = shap.KernelExplainer(
                self.model.predict,
                background[feature_names].values,
            )

            shap_values = explainer.shap_values(row_df[feature_names].values)

            # Handle different SHAP return formats:
            # - Older SHAP / multi-output: returns list of arrays
            # - Newer SHAP / single-output: returns 2D array directly
            if isinstance(shap_values, list):
                # Multi-output case: take first output's values
                shap_values = shap_values[0]

            # Ensure we have a 2D array and extract the single row
            shap_values = np.atleast_2d(shap_values)
            instance_shap = shap_values[0] if shap_values.shape[0] == 1 else shap_values.flatten()

            all_shap_values.append(instance_shap)
            all_base_values.append(
                explainer.expected_value[0]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                and np.asarray(explainer.expected_value).ndim > 0
                and len(explainer.expected_value) == 1
                else explainer.expected_value
            )

        return SHAPResult(
            shap_values=np.array(all_shap_values),
            base_values=np.array(all_base_values).flatten(),
            feature_names=feature_names,
            data=X,
        )

    def explain_instance(
        self,
        instance: pd.Series | pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> SHAPResult:
        """Explain a single instance using series-specific background.

        Args:
            instance: Single instance to explain (Series or single-row DataFrame).
            feature_names: Names of features to include.

        Returns:
            SHAPResult for the single instance.
        """
        if isinstance(instance, pd.Series):
            instance = instance.to_frame().T
            if isinstance(instance.index[0], tuple) and isinstance(
                self.background_data.index, pd.MultiIndex
            ):
                instance.index = pd.MultiIndex.from_tuples(
                    instance.index, names=self.background_data.index.names
                )

        return self.explain(instance, feature_names)

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

    def global_importance(
        self,
        X: pd.DataFrame,
        n_samples: int | None = None,
    ) -> pd.DataFrame:
        """Compute global feature importance from SHAP values.

        Args:
            X: Dataset to compute global importance over.
            n_samples: Number of samples to use. If None, uses all.

        Returns:
            DataFrame with mean absolute SHAP values per feature.
        """
        if n_samples is not None and n_samples < len(X):
            indices = self._rng.choice(len(X), size=n_samples, replace=False)
            X = X.iloc[indices]

        result = self.explain(X)
        return result.mean_abs_shap()
