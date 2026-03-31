"""Partitioners for creating conditional subgroups."""

from timelens.partitioners.manual import ManualPartitioner
from timelens.partitioners.tree import TreePartitioner

__all__ = [
    "ManualPartitioner",
    "TreePartitioner",
]
