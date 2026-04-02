# Partitioners

Partitioners create groups/subsets of data for conditional permutation.

## ManualPartitioner

Use when you have domain knowledge about how series should be grouped.

::: xeries.partitioners.manual.ManualPartitioner
    options:
      show_root_heading: true
      show_source: true

## TreePartitioner

Automatically learns subgroups using a decision tree (cs-PFI algorithm).

::: xeries.partitioners.tree.TreePartitioner
    options:
      show_root_heading: true
      show_source: true

## Base Class

::: xeries.core.base.BasePartitioner
    options:
      show_root_heading: true
      show_source: true
