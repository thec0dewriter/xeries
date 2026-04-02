"""Base adapter interface for forecasting framework integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BaseAdapter(ABC):
    """Abstract base class for forecasting framework adapters.

    Adapters provide a consistent interface for extracting training data
    and making predictions from various forecasting frameworks.
    """

    @abstractmethod
    def get_training_data(self, *args: Any, **kwargs: Any) -> tuple[pd.DataFrame, pd.Series]:
        """Extract training features (X) and target (y) from the forecaster.

        Framework-specific adapters may require extra arguments (e.g. skforecast
        needs the same ``series`` passed to ``fit``).

        Returns:
            Tuple of (X, y) where X is a DataFrame with features and y is the target.
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> NDArray[Any]:
        """Make predictions using the underlying model.

        Args:
            X: Input features DataFrame.

        Returns:
            Array of predictions.
        """
        ...

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Get the names of features used by the model.

        Returns:
            List of feature names.
        """
        ...

    @abstractmethod
    def get_series_column(self) -> str:
        """Get the name of the column/index level containing series identifiers.

        Returns:
            Name of the series identifier column.
        """
        ...
