# Architecture Overview

xeries is organized as a layered explainability toolkit for multi-series forecasting. The architecture now includes feature importance methods, causal analysis, temporal/error analysis, and a unified dashboard orchestration API.

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

- `src/xeries/core/base.py`
- `src/xeries/core/types.py`
- `src/xeries/adapters/base.py`

### 2. Explainer Layer

Implemented explainers:

- `ConditionalPermutationImportance`
- `ConditionalDropImportance`
- `ConditionalSHAP`
- `ConditionalSHAPIQ`
- `CausalFeatureImportance`

Primary files:

- `src/xeries/importance/permutation.py`
- `src/xeries/importance/dropping.py`
- `src/xeries/importance/shap.py`
- `src/xeries/importance/shapiq.py`
- `src/xeries/importance/causal.py`

### 3. Adapter Layer

Implemented adapters:

- `SkforecastAdapter`
- `SklearnAdapter`
- `DartsAdapter`

Primary files:

- `src/xeries/adapters/skforecast.py`
- `src/xeries/adapters/sklearn.py`
- `src/xeries/adapters/darts.py`

### 4. Analysis Layer

Reusable analysis utilities that are independent from any single explainer:

- `ErrorAnalyzer` (global/per-series/per-window errors)
- `TemporalImportance` (windowed importance execution)
- ranking comparison utilities
- bootstrap significance utilities

Primary files:

- `src/xeries/analysis/error.py`
- `src/xeries/analysis/temporal.py`
- `src/xeries/analysis/comparison.py`
- `src/xeries/analysis/significance.py`

### 5. Dashboard Layer

Builder-style orchestration API:

- `Dashboard`
- components: interpretability, error analysis, causal, interactions
- `DashboardResult`
- HTML report generation and browser rendering

Primary files:

- `src/xeries/dashboard/core.py`
- `src/xeries/dashboard/components/`
- `src/xeries/dashboard/results.py`
- `src/xeries/dashboard/report.py`

### 6. Visualization Layer

- `plot_importance_bar`
- `plot_importance_heatmap`
- `plot_shap_summary`
- `plot_shap_bar`
- `plot_importance_comparison`

Primary file:

- `src/xeries/visualization/plots.py`

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
xeries/
├── core/            # base abstractions and result types
├── partitioners/    # grouping strategies for conditional methods
├── adapters/        # framework adapters (skforecast, sklearn, darts)
├── importance/      # explainers (permutation, dropping, SHAP, SHAP-IQ, causal)
├── analysis/        # error/temporal/comparison/significance utilities
├── dashboard/       # orchestration API + components + report rendering
└── visualization/   # plotting helpers
```
