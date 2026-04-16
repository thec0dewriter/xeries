"""SHAP implementations for time series models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from xeries.core.base import AttributionExplainer
from xeries.core.types import ModelProtocol, SHAPResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

ExplainerType = Literal["tree", "linear", "kernel", "deep", "auto"]


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
                if isinstance(explainer.expected_value, list | np.ndarray)
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


class BatchSHAP:
    """Batch SHAP computation with external explainer support.

    Computes SHAP values for an entire dataset at once for efficiency,
    supporting various SHAP explainer types (TreeExplainer, LinearExplainer, etc.).
    Tracks series membership for later hierarchical aggregation.

    This class follows the standard SHAP library pattern:
        >>> explainer = shap.TreeExplainer(model)
        >>> shap_values = explainer(X)  # Batch computation

    Example with external explainer:
        >>> import shap
        >>> tree_exp = shap.TreeExplainer(model)
        >>> batch_shap = BatchSHAP(explainer=tree_exp, series_col='level')
        >>> result = batch_shap.explain(X)

    Example with auto-created explainer:
        >>> batch_shap = BatchSHAP(model=model, explainer_type='tree')
        >>> result = batch_shap.explain(X)

    Example with hierarchical aggregation:
        >>> from xeries.hierarchy import HierarchyDefinition, HierarchicalAggregator
        >>> hierarchy = HierarchyDefinition(
        ...     levels=['state', 'store'],
        ...     columns=['state_id', 'store_id']
        ... )
        >>> aggregator = HierarchicalAggregator(hierarchy)
        >>> hierarchical_result = aggregator.aggregate_shap(result, X)
    """

    SUPPORTED_EXPLAINER_TYPES = ("tree", "linear", "kernel", "deep", "auto")

    def __init__(
        self,
        model: Any = None,
        explainer: Any = None,
        explainer_type: ExplainerType = "tree",
        background_data: pd.DataFrame | np.ndarray | None = None,
        series_col: str | None = None,
        **explainer_kwargs: Any,
    ) -> None:
        """Initialize the batch SHAP explainer.

        Must provide either `explainer` (pre-created SHAP explainer) or `model`
        (to auto-create explainer).

        Args:
            model: A model with a predict method. Required if `explainer` is None.
            explainer: Pre-created SHAP explainer (TreeExplainer, LinearExplainer, etc.).
                If provided, `model` and `explainer_type` are ignored.
            explainer_type: Type of explainer to create if `explainer` is None.
                Options: 'tree', 'linear', 'kernel', 'deep', 'auto'.
            background_data: Background data for KernelExplainer or LinearExplainer.
                Required for 'kernel' and 'linear' types.
            series_col: Column name containing series identifiers for tracking.
                If None, series tracking is disabled.
            **explainer_kwargs: Additional keyword arguments passed to the SHAP explainer.

        Raises:
            ValueError: If neither model nor explainer is provided.
            ValueError: If explainer_type is invalid.
            ImportError: If shap package is not installed.
        """
        try:
            import shap

            self._shap = shap
        except ImportError as e:
            raise ImportError(
                "shap package is required for BatchSHAP. Install it with: pip install shap"
            ) from e

        if explainer is None and model is None:
            raise ValueError("Either 'model' or 'explainer' must be provided.")

        if explainer_type not in self.SUPPORTED_EXPLAINER_TYPES:
            raise ValueError(
                f"Invalid explainer_type '{explainer_type}'. "
                f"Supported: {self.SUPPORTED_EXPLAINER_TYPES}"
            )

        self.model = model
        self.series_col = series_col
        self.background_data = background_data
        self.explainer_type = explainer_type
        self._explainer_kwargs = explainer_kwargs

        if explainer is not None:
            self.explainer = explainer
        else:
            self.explainer = self._create_explainer(model, explainer_type, background_data)

    def _create_explainer(
        self,
        model: Any,
        explainer_type: ExplainerType,
        background_data: pd.DataFrame | np.ndarray | None,
    ) -> Any:
        """Create a SHAP explainer based on the specified type.

        Args:
            model: The model to explain.
            explainer_type: Type of explainer to create.
            background_data: Background data for explainers that require it.

        Returns:
            SHAP explainer instance.

        Raises:
            ValueError: If background_data is required but not provided.
        """
        shap = self._shap

        if explainer_type == "auto":
            explainer_type = self._detect_explainer_type(model)

        if explainer_type == "tree":
            return shap.TreeExplainer(model, **self._explainer_kwargs)
        elif explainer_type == "linear":
            if background_data is None:
                raise ValueError("background_data is required for LinearExplainer.")
            return shap.LinearExplainer(model, background_data, **self._explainer_kwargs)
        elif explainer_type == "kernel":
            if background_data is None:
                raise ValueError("background_data is required for KernelExplainer.")
            bg_data = (
                background_data.values
                if isinstance(background_data, pd.DataFrame)
                else background_data
            )
            return shap.KernelExplainer(model.predict, bg_data, **self._explainer_kwargs)
        elif explainer_type == "deep":
            if background_data is None:
                raise ValueError("background_data is required for DeepExplainer.")
            return shap.DeepExplainer(model, background_data, **self._explainer_kwargs)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

    def _detect_explainer_type(self, model: Any) -> ExplainerType:
        """Auto-detect the appropriate explainer type for a model.

        Args:
            model: The model to detect explainer type for.

        Returns:
            Detected explainer type.
        """
        model_type = type(model).__name__.lower()
        module = type(model).__module__.lower()

        tree_indicators = [
            "lightgbm",
            "xgboost",
            "catboost",
            "randomforest",
            "gradientboosting",
            "extratrees",
            "decisiontree",
            "lgbm",
            "xgb",
            "cb",
        ]

        linear_indicators = [
            "linear",
            "logistic",
            "ridge",
            "lasso",
            "elasticnet",
        ]

        deep_indicators = [
            "keras",
            "tensorflow",
            "torch",
            "neural",
            "mlp",
        ]

        combined = model_type + " " + module

        for indicator in tree_indicators:
            if indicator in combined:
                return "tree"

        for indicator in linear_indicators:
            if indicator in combined:
                return "linear"

        for indicator in deep_indicators:
            if indicator in combined:
                return "deep"

        return "kernel"

    def _get_series_ids(self, X: pd.DataFrame) -> pd.Series | None:
        """Extract series identifiers from DataFrame.

        Args:
            X: Input DataFrame.

        Returns:
            Series of identifiers, or None if series_col is not set.
        """
        if self.series_col is None:
            return None

        if isinstance(X.index, pd.MultiIndex) and self.series_col in X.index.names:
            return X.index.get_level_values(self.series_col).to_series(index=X.index)

        if self.series_col in X.columns:
            return X[self.series_col].reset_index(drop=True)

        if self.series_col == "level" and "_level_skforecast" in X.columns:
            return X["_level_skforecast"].reset_index(drop=True)

        raise KeyError(
            f"Series column '{self.series_col}' not found in DataFrame columns or index"
        )

    def explain(
        self,
        X: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> SHAPResult:
        """Compute SHAP values in batch for entire dataset.

        Args:
            X: Dataset to explain with shape (n_samples, n_features).
            feature_names: Names of features to use. If None, all numeric columns
                are used (excluding series_col if present).

        Returns:
            SHAPResult containing SHAP values, base values, and series_ids.
        """
        if feature_names is None:
            feature_names = self._infer_feature_names(X)

        series_ids = self._get_series_ids(X)

        X_features = X[feature_names]

        explanation = self.explainer(X_features)

        shap_values = self._extract_shap_values(explanation)
        base_values = self._extract_base_values(explanation, len(X))

        return SHAPResult(
            shap_values=shap_values,
            base_values=base_values,
            feature_names=feature_names,
            data=X,
            series_ids=series_ids,
        )

    def _infer_feature_names(self, X: pd.DataFrame) -> list[str]:
        """Infer feature names from DataFrame, excluding series column.

        Args:
            X: Input DataFrame.

        Returns:
            List of feature column names.
        """
        exclude_cols = set()
        if self.series_col is not None:
            exclude_cols.add(self.series_col)
            if self.series_col == "level":
                exclude_cols.add("_level_skforecast")

        return [col for col in X.columns if col not in exclude_cols]

    def _extract_shap_values(self, explanation: Any) -> NDArray[np.floating[Any]]:
        """Extract SHAP values array from explanation object.

        Args:
            explanation: SHAP Explanation object.

        Returns:
            2D array of SHAP values with shape (n_samples, n_features).
        """
        values = explanation.values

        if isinstance(values, list):
            values = values[0]

        if values.ndim == 3:
            values = values[:, :, 0]

        return np.asarray(values, dtype=np.float64)

    def _extract_base_values(self, explanation: Any, n_samples: int) -> NDArray[np.floating[Any]]:
        """Extract base values from explanation object.

        Args:
            explanation: SHAP Explanation object.
            n_samples: Number of samples (used for broadcasting scalar base value).

        Returns:
            1D array of base values with shape (n_samples,).
        """
        base_values = explanation.base_values

        if isinstance(base_values, list | np.ndarray):
            base_values = np.asarray(base_values)
            if base_values.ndim == 0:
                base_values = np.full(n_samples, float(base_values))
            elif base_values.ndim == 2:
                base_values = base_values[:, 0]
        else:
            base_values = np.full(n_samples, float(base_values))

        return base_values.astype(np.float64)

    def global_importance(self, result: SHAPResult | None = None) -> pd.DataFrame:
        """Compute global feature importance from SHAP values.

        Args:
            result: Optional SHAPResult to compute importance from.
                If None, must call explain() first.

        Returns:
            DataFrame with mean absolute SHAP values per feature.

        Raises:
            ValueError: If result is None.
        """
        if result is None:
            raise ValueError("result must be provided. Call explain(X) first.")
        return result.mean_abs_shap()
