# Architecture Overview

The Time-Conditional Permutation Feature Importance (timelens) library is designed with a modular, extensible architecture specifically for time series explainability methods. This section describes the high-level structure and organization.

## Quick Navigation

- **[API Structure & Inheritance](api_structure.md)** - Detailed documentation of all classes, inheritance patterns, and design patterns
- **[Mermaid Diagrams](inheritance_diagrams.md)** - Visual representations of class hierarchies and data flow

## Architecture at a Glance

The library consists of **4 core architectural layers**:

### 1. **Foundation Layer: Core Base Classes**
- `BasePartitioner` - Contract for data partitioning strategies
- `BaseExplainer` - Base for all feature importance calculators
  - `MetricBasedExplainer` - For performance-drop strategies (PFI)
  - `AttributionExplainer` - For direct feature attribution (SHAP, SHAP-IQ)
  - `CausalExplainer` - Integration point for causal models (Future)
- `BaseAdapter` - Interface for framework integration

**Location**: `src/timelens/core/base.py`

### 2. **Strategy Layer: Implementations**

**Partitioners** (`src/timelens/partitioners/`):
- `ManualPartitioner` - User-defined group mappings
- `TreePartitioner` - Automated tree-based subgroup discovery

**Explainers** (`src/timelens/importance/`):
- `ConditionalPermutationImportance` - Main algorithm with flexible partitioning
- `ConditionalSHAP` - Series-aware SHAP values for multi-series models

**Adapters** (`src/timelens/adapters/`):
- `SkforecastAdapter` - Integration with skforecast forecasting models

### 3. **Visualization Layer**
- `plot_importance_bar()` - Horizontal bar chart of feature importance
- `plot_importance_heatmap()` - Multi-condition heatmap comparison

**Location**: `src/timelens/visualization/plots.py`

### 4. **Type System**
- `BaseResult` - Base protocol for output results
- `FeatureImportanceResult` - Container for PFI scores
- `SHAPResult` - Container for SHAP values
- Protocol types for model contracts

**Location**: `src/timelens/core/types.py`

---

## Key Design Principles

### Composition Over Inheritance
Explainers *compose* partitioners rather than inheriting from them:
```python
explainer = ConditionalPermutationImportance(partitioner=TreePartitioner(...))
```

### Strategy Pattern
Interchangeable partitioning strategies (auto vs. manual) provide runtime flexibility:
```python
# Strategy 1: Automatic
ConditionalPermutationImportance(strategy='auto')

# Strategy 2: Manual
ConditionalPermutationImportance(strategy='manual', partitioner=manual_partitioner)
```

### Adapter Pattern
Framework-specific adapters provide a unified interface:
```python
adapter = SkforecastAdapter(forecaster, series=data)
X, y = adapter.get_training_data()
```

---

## Typical Workflow

```
1. Load/Train Model
   └─ Forecaster, classifier, or any model with predict()

2. Extract Training Data (Adapter)
   └─ SkforecastAdapter.get_training_data()
   └─ Output: X (features), y (target)

3. Create Importance Explainer
   └─ ConditionalPermutationImportance / ConditionalSHAP
   └─ Specify: strategy, metric, partitioner, etc.

4. Compute Importance
   └─ explainer.explain(X, y, ...)
   └─ Output: Explanation Result (e.g., FeatureImportanceResult)

5. Visualize Results
   └─ plot_importance_bar(result)
   └─ plot_importance_heatmap(results_dict)
```

---

## Module Dependencies

```
timelens/
├── core/
│   ├── base.py          → Abstract base classes (BaseExplainer, MetricBasedExplainer, AttributionExplainer)
│   └── types.py         → Type definitions, protocols
│
├── partitioners/
│   ├── base.py (none)   → Inherits from core.base
│   ├── manual.py        → ManualPartitioner → BasePartitioner
│   └── tree.py          → TreePartitioner → BasePartitioner (+ sklearn)
│
├── importance/
│   ├── permutation.py   → ConditionalPermutationImportance → MetricBasedExplainer
│   └── shap.py          → ConditionalSHAP → AttributionExplainer
│
├── adapters/
│   ├── base.py          → BaseAdapter
│   └── skforecast.py    → SkforecastAdapter → BaseAdapter
│
└── visualization/
    └── plots.py         → Utility functions (no classes)
```

---

## For More Details

See [API Structure & Inheritance](api_structure.md) for:
- Complete class descriptions
- Method signatures and responsibilities
- Integration examples
- Pattern explanations

See [Mermaid Diagrams](inheritance_diagrams.md) for:
- Visual class hierarchies
- Data flow diagrams
- Module organization
- Composition relationships
