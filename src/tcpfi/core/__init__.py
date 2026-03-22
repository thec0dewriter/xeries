"""Core module containing base classes and type definitions."""

from tcpfi.core.base import BaseExplainer, BasePartitioner
from tcpfi.core.types import (
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
