"""Manual dictionary-based partitioner for conditional permutation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from xeries.core.base import BasePartitioner


class ManualPartitioner(BasePartitioner):
    """Partitioner using a user-defined mapping dictionary.

    This partitioner assigns samples to groups based on a predefined mapping
    from series identifiers (or other categorical values) to group labels.
    Useful when domain knowledge suggests natural groupings.

    Example:
        >>> mapping = {'MT_001': 'group_A', 'MT_002': 'group_B', 'MT_003': 'group_A'}
        >>> partitioner = ManualPartitioner(mapping, series_col='level')
        >>> groups = partitioner.fit_get_groups(X, feature='lag_1')
    """

    def __init__(
        self,
        mapping: dict[Any, Any],
        series_col: str = "level",
    ) -> None:
        """Initialize the manual partitioner.

        Args:
            mapping: Dictionary mapping series identifiers to group labels.
            series_col: Name of the column or index level containing series IDs.
        """
        self.mapping = mapping
        self.series_col = series_col
        self._fitted = False
        self._group_encoder: dict[Any, int] = {}

    def fit(self, X: pd.DataFrame, feature: str) -> ManualPartitioner:
        """Fit the partitioner (encodes group labels to integers).

        Args:
            X: Input features DataFrame.
            feature: The feature to condition on (not used for manual partitioner).

        Returns:
            Self for method chaining.
        """
        unique_groups = sorted(set(self.mapping.values()))
        self._group_encoder = {g: i for i, g in enumerate(unique_groups)}
        self._fitted = True
        return self

    def get_groups(self, X: pd.DataFrame) -> NDArray[np.intp]:
        """Get group labels for each sample based on the mapping.

        Args:
            X: Input features DataFrame with series identifiers.

        Returns:
            Array of integer group labels.

        Raises:
            ValueError: If partitioner has not been fitted.
            KeyError: If series_col is not found in X.
        """
        if not self._fitted:
            raise ValueError("Partitioner must be fitted before calling get_groups")

        series_ids = self._get_series_ids(X)
        group_labels = series_ids.map(self.mapping)

        if group_labels.isna().any():
            missing = series_ids[group_labels.isna()].unique()
            raise ValueError(f"Series IDs not found in mapping: {missing.tolist()}")

        encoded = group_labels.map(self._group_encoder)
        return encoded.to_numpy().astype(np.intp)

    def _get_series_ids(self, X: pd.DataFrame) -> pd.Series:
        """Extract series identifiers from DataFrame."""
        if isinstance(X.index, pd.MultiIndex) and self.series_col in X.index.names:
            return X.index.get_level_values(self.series_col).to_series(index=X.index)

        if self.series_col in X.columns:
            return X[self.series_col]

        raise KeyError(f"Series column '{self.series_col}' not found in DataFrame columns or index")

    @property
    def n_groups(self) -> int:
        """Return the number of unique groups."""
        return len(self._group_encoder)
