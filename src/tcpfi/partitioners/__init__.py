"""Partitioners for creating conditional subgroups."""

from tcpfi.partitioners.manual import ManualPartitioner
from tcpfi.partitioners.tree import TreePartitioner

__all__ = [
    "ManualPartitioner",
    "TreePartitioner",
]
