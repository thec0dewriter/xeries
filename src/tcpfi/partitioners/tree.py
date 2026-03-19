"""Tree-based partitioner for automated conditional subgroup discovery (cs-PFI)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from tcpfi.core.base import BasePartitioner


class TreePartitioner(BasePartitioner):
    """Partitioner using decision tree leaf nodes for subgroup discovery.

    This implements the Conditional Subgroup Permutation Feature Importance
    (cs-PFI) algorithm. A decision tree is trained to predict the feature
    of interest using all other features. The leaf nodes of this tree
    define homogeneous subgroups for conditional permutation.

    Example:
        >>> partitioner = TreePartitioner(max_depth=4, min_samples_leaf=0.05)
        >>> groups = partitioner.fit_get_groups(X, feature='lag_1')
    """

    def __init__(
        self,
        max_depth: int | None = 4,
        min_samples_leaf: int | float = 0.05,
        series_col: str | None = "level",
        random_state: int | None = None,
    ) -> None:
        """Initialize the tree partitioner.

        Args:
            max_depth: Maximum depth of the decision tree.
            min_samples_leaf: Minimum samples required in a leaf node.
                Can be int (absolute) or float (fraction of total samples).
            series_col: Column/index level containing series IDs to encode.
                Set to None to skip series encoding.
            random_state: Random seed for reproducibility.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.series_col = series_col
        self.random_state = random_state

        self._tree: DecisionTreeRegressor | None = None
        self._encoder: OneHotEncoder | None = None
        self._feature: str | None = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, feature: str) -> TreePartitioner:
        """Fit the decision tree to predict the feature of interest.

        Args:
            X: Input features DataFrame.
            feature: The feature to condition on (will be predicted by tree).

        Returns:
            Self for method chaining.
        """
        self._feature = feature

        y_tree = X[feature].values
        X_tree = self._prepare_tree_features(X, feature)

        self._tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self._tree.fit(X_tree, y_tree)
        self._fitted = True

        return self

    def get_groups(self, X: pd.DataFrame) -> NDArray[np.intp]:
        """Get leaf node indices as group labels.

        Args:
            X: Input features DataFrame.

        Returns:
            Array of leaf node indices (group labels).

        Raises:
            ValueError: If partitioner has not been fitted.
        """
        if not self._fitted or self._tree is None or self._feature is None:
            raise ValueError("Partitioner must be fitted before calling get_groups")

        X_tree = self._prepare_tree_features(X, self._feature)
        return self._tree.apply(X_tree).astype(np.intp)

    def _prepare_tree_features(
        self,
        X: pd.DataFrame,
        feature: str,
    ) -> NDArray[np.floating[Any]]:
        """Prepare features for the decision tree.

        Args:
            X: Input features DataFrame.
            feature: The feature to exclude (target for tree).

        Returns:
            Prepared feature array for tree training/prediction.
        """
        X_reset = X.reset_index() if isinstance(X.index, pd.MultiIndex) else X.copy()
        X_tree = X_reset.drop(columns=[feature], errors="ignore")

        if "date" in X_tree.columns:
            X_tree = X_tree.drop(columns=["date"])

        if self.series_col and self.series_col in X_tree.columns:
            series_data = X_tree[[self.series_col]]

            if self._encoder is None:
                self._encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded = self._encoder.fit_transform(series_data)
            else:
                encoded = self._encoder.transform(series_data)

            encoded_df = pd.DataFrame(
                encoded,
                columns=self._encoder.get_feature_names_out([self.series_col]),
                index=X_tree.index,
            )
            X_tree = pd.concat(
                [X_tree.drop(columns=[self.series_col]), encoded_df],
                axis=1,
            )

        numeric_cols = X_tree.select_dtypes(include=[np.number]).columns
        return X_tree[numeric_cols].to_numpy()

    @property
    def n_groups(self) -> int:
        """Return the number of leaf nodes (groups)."""
        if self._tree is None:
            return 0
        return int(self._tree.get_n_leaves())

    @property
    def tree(self) -> DecisionTreeRegressor | None:
        """Return the fitted decision tree."""
        return self._tree
