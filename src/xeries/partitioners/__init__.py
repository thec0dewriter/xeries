"""Partitioners for creating conditional subgroups."""

from xeries.partitioners.manual import ManualPartitioner
from xeries.partitioners.tree import TreePartitioner

__all__ = [
    "ManualPartitioner",
    "TreePartitioner",
]
