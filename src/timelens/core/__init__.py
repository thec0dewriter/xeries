"""Core module containing base classes and type definitions."""

from timelens.core.base import BaseExplainer, BasePartitioner
from timelens.core.types import (
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
