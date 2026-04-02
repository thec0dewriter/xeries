# API Structure & Inheritance

This document describes the current API surface and inheritance relationships in xeries.

## Core Abstractions

### xeries.core.base

- `BasePartitioner`: contract for subgroup generation (`fit`, `get_groups`, `fit_get_groups`).
- `BaseExplainer`: common `explain(...)` contract.
- `MetricBasedExplainer`: metric-driven explainers (`mse`, `mae`, `rmse`, or callable).
- `AttributionExplainer`: attribution explainers with background data support.
- `CausalExplainer`: causal explainers with treatment features, graph metadata, and series context.

### xeries.core.types

- `BaseResult`
- `FeatureImportanceResult`
- `SHAPResult`
- `CausalResult`
- `RefutationResult`
- protocol aliases (`ModelProtocol`, `MetricFunction`, `ArrayLike`, etc.)

## Partitioners

### xeries.partitioners.manual.ManualPartitioner

- manual mapping from series/category values to groups.

### xeries.partitioners.tree.TreePartitioner

- cs-PFI style automatic grouping using tree-based structure.

## Explainers

### Metric-based explainers

- `ConditionalPermutationImportance` (`MetricBasedExplainer`)
- `ConditionalDropImportance` (`MetricBasedExplainer`)

### Attribution explainers

- `ConditionalSHAP` (`AttributionExplainer`)
- `ConditionalSHAPIQ` (`AttributionExplainer`)
- `SHAPIQResult` (result object for interaction analysis)

### Causal explainers

- `CausalFeatureImportance` (`CausalExplainer`)
  - supports estimators: `causal_forest`, `linear_dml`, `dynamic_dml`
  - supports graph discovery and refutation helpers

## Adapters

All adapters inherit `BaseAdapter` and provide a consistent tabular contract:

- `SkforecastAdapter`
- `SklearnAdapter`
- `DartsAdapter`

Required methods:

- `get_training_data(...) -> (X, y)`
- `predict(X)`
- `get_feature_names()`
- `get_series_column()`

## Analysis Utilities

### xeries.analysis.error.ErrorAnalyzer

- global metrics
- per-series metrics
- fixed-window temporal metrics

### xeries.analysis.temporal.TemporalImportance

- run an explainer across windows and return window-feature importance frames.

### xeries.analysis.comparison.compare_rankings

- pairwise ranking agreement matrix (Kendall-style agreement).

### xeries.analysis.significance

- `bootstrap_interval`
- `estimate_significance`

## Dashboard API

### xeries.dashboard.core.Dashboard

Builder methods:

- `add_interpretability(...)`
- `add_error_analysis(...)`
- `add_causal_analysis(...)`
- `add_interactions(...)`

Execution/output methods:

- `compute() -> DashboardResult`
- `plot_all()`
- `generate_report(path)`
- `generate_scorecard(path)`
- `show()`

### Components

- `InterpretabilityComponent`
- `ErrorAnalysisComponent`
- `CausalComponent`
- `InteractionComponent`

### Results and report

- `DashboardResult` with `compare_rankings()` and `summary()`
- HTML report rendering via Jinja2 in `dashboard/report.py`

## Visualization API

### xeries.visualization.plots

- `plot_importance_bar`
- `plot_importance_heatmap`
- `plot_shap_summary`
- `plot_shap_bar`
- `plot_importance_comparison`

## Integration Patterns

- Strategy pattern: partitioners and method selection.
- Adapter pattern: framework-specific wrappers.
- Composition: dashboard orchestrates explainers and analyzers rather than duplicating logic.
