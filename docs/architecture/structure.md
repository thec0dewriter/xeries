# Architecture Overview

timelens is organized as a layered explainability toolkit for multi-series forecasting. The architecture now includes feature importance methods, causal analysis, temporal/error analysis, and a unified dashboard orchestration API.

## Quick Navigation

- [API Structure & Inheritance](api_structure.md)
- [Mermaid Diagrams](inheritance_diagrams.md)

## Architecture at a Glance

### 1. Foundation Layer

- `BasePartitioner`
- `BaseExplainer` and specialized bases:
  - `MetricBasedExplainer`
  - `AttributionExplainer`
  - `CausalExplainer`
- `BaseAdapter`
- typed result objects in `core/types.py`

Primary files:

- `src/timelens/core/base.py`
- `src/timelens/core/types.py`
- `src/timelens/adapters/base.py`

### 2. Explainer Layer

Implemented explainers:

- `ConditionalPermutationImportance`
- `ConditionalDropImportance`
- `ConditionalSHAP`
- `ConditionalSHAPIQ`
- `CausalFeatureImportance`

Primary files:

- `src/timelens/importance/permutation.py`
- `src/timelens/importance/dropping.py`
- `src/timelens/importance/shap.py`
- `src/timelens/importance/shapiq.py`
- `src/timelens/importance/causal.py`

### 3. Adapter Layer

Implemented adapters:

- `SkforecastAdapter`
- `SklearnAdapter`
- `DartsAdapter`

Primary files:

- `src/timelens/adapters/skforecast.py`
- `src/timelens/adapters/sklearn.py`
- `src/timelens/adapters/darts.py`

### 4. Analysis Layer

Reusable analysis utilities that are independent from any single explainer:

- `ErrorAnalyzer` (global/per-series/per-window errors)
- `TemporalImportance` (windowed importance execution)
- ranking comparison utilities
- bootstrap significance utilities

Primary files:

- `src/timelens/analysis/error.py`
- `src/timelens/analysis/temporal.py`
- `src/timelens/analysis/comparison.py`
- `src/timelens/analysis/significance.py`

### 5. Dashboard Layer

Builder-style orchestration API:

- `Dashboard`
- components: interpretability, error analysis, causal, interactions
- `DashboardResult`
- HTML report generation and browser rendering

Primary files:

- `src/timelens/dashboard/core.py`
- `src/timelens/dashboard/components/`
- `src/timelens/dashboard/results.py`
- `src/timelens/dashboard/report.py`

### 6. Visualization Layer

- `plot_importance_bar`
- `plot_importance_heatmap`
- `plot_shap_summary`
- `plot_shap_bar`
- `plot_importance_comparison`

Primary file:

- `src/timelens/visualization/plots.py`

## Design Principles

### Composition over inheritance

Explainers compose partitioners and models rather than inheriting implementation behavior from them.

### Strategy-driven execution

Users can switch partitioning and method choices with runtime parameters (`strategy`, `methods`, estimator selections).

### Adapter abstraction

Framework-specific model interfaces are wrapped into a common contract (`get_training_data`, `predict`, metadata helpers).

### Orchestration isolation

`Dashboard` orchestrates existing explainers and analysis helpers instead of re-implementing ML logic.

## Typical Workflows

### Direct explainer workflow

1. Build adapter for your model/framework.
2. Extract `X`, `y`.
3. Run one explainer.
4. Visualize or export result tables.

### Unified dashboard workflow

1. Create `Dashboard(model, X, y, series_col=...)`.
2. Add components with `add_*` builder methods.
3. Execute `compute()`.
4. Inspect `DashboardResult`, compare rankings, and generate HTML reports.

## Module Map

```text
timelens/
├── core/            # base abstractions and result types
├── partitioners/    # grouping strategies for conditional methods
├── adapters/        # framework adapters (skforecast, sklearn, darts)
├── importance/      # explainers (permutation, dropping, SHAP, SHAP-IQ, causal)
├── analysis/        # error/temporal/comparison/significance utilities
├── dashboard/       # orchestration API + components + report rendering
└── visualization/   # plotting helpers
```
