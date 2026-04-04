"""Adapters for integrating with forecasting frameworks."""

from xeries.adapters.base import BaseAdapter
# from xeries.adapters.darts import DartsAdapter
from xeries.adapters.skforecast import SkforecastAdapter, from_skforecast
from xeries.adapters.sklearn import SklearnAdapter

__all__ = [
    "BaseAdapter",
    # "DartsAdapter",
    "SkforecastAdapter",
    "SklearnAdapter",
    "from_skforecast",
]
