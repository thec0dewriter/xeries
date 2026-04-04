"""Core module containing base classes and type definitions."""

from xeries.core.base import BaseExplainer, BasePartitioner
from xeries.core.types import (
    ArrayLike,
    FeatureImportanceResult,
    GroupLabels,
    ModelProtocol,
)

__all__ = [
    "ArrayLike",
    "BaseExplainer",
    "BasePartitioner",
    "FeatureImportanceResult",
    "GroupLabels",
    "ModelProtocol",
]
