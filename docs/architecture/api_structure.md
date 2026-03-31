# API Structure & Inheritance

The Time-Conditional PFI (timelens) library follows a modular architecture with clear separation of concerns. The API is built around four main components connected through abstract base classes and composition patterns.

## Core Components

### 1. **Core Base Classes** (`timelens.core.base`)

The foundation of the library consists of several abstract base classes defining the core hierarchy:

#### `BasePartitioner` (ABC)
**Purpose**: Define the interface for data partitioning strategies  
**Key Methods**:
- `fit(X, feature)` - Learn partitioning based on feature data
- `get_groups(X)` - Return group assignments for samples
- `fit_get_groups(X, feature)` - Convenience method combining both steps

**Use Case**: Different strategies for dividing data into homogeneous subgroups for conditional permutation

#### `BaseExplainer` (ABC)
**Purpose**: Root base class for all explainers  
**Responsibilities**:
- Define the universal `explain(X, **kwargs)` interface

#### `MetricBasedExplainer` (Inherits `BaseExplainer`)
**Purpose**: Base for performance drop methods like permutation importance
**Responsibilities**:
- Initialize model and metric (MSE, MAE, RMSE, or custom callable)
- Provide metric resolution logic

#### `AttributionExplainer` (Inherits `BaseExplainer`)
**Purpose**: Base for direct feature attribution methods like SHAP and SHAP-IQ
**Responsibilities**:
- Initialize model and background data for scaling and baselines

#### `CausalExplainer` (Inherits `BaseExplainer`)
**Purpose**: Future extension for causal discovery and DAG-based explainability

---

### 2. **Partitioners** (`timelens.partitioners`)

Concrete implementations of `BasePartitioner` for different partitioning strategies:

```
BasePartitioner
├── ManualPartitioner
│   └── User-defined mapping (series_id → group_label)
│   └── Best for: Domain knowledge-based groupings
│   └── Input: mapping dict, series_col name
│
└── TreePartitioner
    └── Automated tree-based discovery (cs-PFI algorithm)
    └── Best for: Automatic subgroup detection
    └── Input: max_depth, min_samples_leaf, random_state
    └── Uses: DecisionTreeRegressor from scikit-learn
    └── Features: OneHotEncoding of series identifiers
```

#### `ManualPartitioner`
- Maps series identifiers or categorical values to predefined groups
- Useful when domain experts know natural groupings
- Encodes group labels to integers for efficiency

#### `TreePartitioner`  
- Implements Conditional Subgroup PFI (cs-PFI) algorithm
- Trains a decision tree to predict the feature of interest
- Leaf nodes become group boundaries
- Auto-detects series column (_level_skforecast or level)
- Supports fractional min_samples_leaf (e.g., 0.05 = 5%)

---

### 3. **Explainers** (`timelens.importance`)

Feature importance calculators with different methodologies:

```
BaseExplainer
├── MetricBasedExplainer
│   └── ConditionalPermutationImportance
│       ├── Strategy: 'auto' → TreePartitioner
│       ├── Strategy: 'manual' → User-provided or ManualPartitioner
│       ├── Supports: Parallel computation (n_jobs)
│       ├── Supports: Multiple repeats for stability (n_repeats)
│       └── Output: FeatureImportanceResult
│
├── AttributionExplainer
│   └── ConditionalSHAP
│       ├── Uses: Series-specific background data
│       ├── Uses: KernelSHAP under the hood
│       ├── Best for: Multi-series forecasting models
│       └── Output: SHAPResult
│
└── CausalExplainer (Future Extensibility)
```

#### `ConditionalPermutationImportance` (inherits from `MetricBasedExplainer`)
**Algorithm**: 
1. Compute baseline predictions and metric
2. For each feature, shuffle values only within subgroups (defined by partitioner)
3. Measure metric increase from baseline
4. Repeat n_repeats times for stability

**Parameters**:
- `strategy`: 'auto' (tree-based) or 'manual' (user-defined groups)
- `partitioner`: Custom partitioner instance (optional)
- `n_repeats`: Number of repetitions for each feature
- `n_jobs`: Parallel jobs (-1 for all cores)
- Inherits: model, metric, random_state from BaseExplainer

**Output**: `FeatureImportanceResult` containing importance scores and standard deviations

#### `ConditionalSHAP` (inherits `AttributionExplainer`)
**Algorithm**: SHAP values using series-specific background data
**Rationale**: For multi-series models, uses background data from the same series as the instance being explained
**Key Features**:
- Prepares per-series background datasets
- Applies KernelSHAP with series-aware baselines
- Auto-detects series column

**Output**: `SHAPResult` with mean absolute SHAP values

---

### 4. **Adapters** (`timelens.adapters`)

Framework-specific adapters for seamless model integration:

```
BaseAdapter (ABC)
└── SkforecastAdapter
    ├── Supports: ForecasterRecursiveMultiSeries
    ├── Supports: Legacy stacked layouts (MultiIndex)
    ├── Auto-detects: Series encoding (_level_skforecast or 'level')
    └── Handles: Training data extraction + caching
```

#### `BaseAdapter` (ABC)
**Purpose**: Unified interface for different forecasting frameworks  
**Key Methods**:
- `get_training_data(*args, **kwargs)` → tuple[pd.DataFrame, pd.Series]
- `predict(X)` → NDArray - Make predictions
- `get_feature_names()` → list[str] - Feature column names
- `get_series_column()` → str - Series identifier column name

#### `SkforecastAdapter`
**Supported Models**: `ForecasterRecursiveMultiSeries` (skforecast 0.21+)  
**Responsibilities**:
- Extract training matrix from fitted forecaster
- Handle both wide (dict) and stacked (MultiIndex) formats
- Support exogenous variables (exog)
- Cache training data for repeated calls

**Key Features**:
- Detects series encoding format automatically
- Handles legacy MultiIndex layouts
- `get_series_ids()` returns decoded series names
- Validates forecaster type on init

---

### 5. **Visualization** (`timelens.visualization`)

Utility functions for plotting importance results (not class-based):

```
plot_importance_bar(result, max_features=20, ax=None, ...)
├── Horizontal bar chart with optional error bars
├── Shows std deviation if available
└── Returns: (Figure, Axes)

plot_importance_heatmap(results, features=None, ax=None, ...)
├── Heatmap comparing importance across conditions
├── Takes dict[condition → FeatureImportanceResult]
├── Returns: (Figure, Axes)
└── Supports: Annotations and custom colormaps
```

---

## Data Flow & Integration

### Typical Workflow

```
1. Framework Integration
   └─ SkforecastAdapter → Extract training data from forecaster

2. Partitioning Strategy
   └─ TreePartitioner.fit_get_groups(X, feature)
   └─ OR ManualPartitioner with predefined mapping

3. Importance Calculation
   └─ ConditionalPermutationImportance.explain(X, y, features)
   └─ Uses partitioner internally for each feature
   └─ Returns: FeatureImportanceResult

4. Visualization
   └─ plot_importance_bar(result)
   └─ Optional: plot_importance_heatmap(multiple_results)
```

### Example Integration

```python
from skforecast.recursive import ForecasterRecursiveMultiSeries
from sklearn.ensemble import RandomForestRegressor
from timelens.adapters import SkforecastAdapter
from timelens.importance import ConditionalPermutationImportance
from timelens.visualization import plot_importance_bar

# 1. Train forecaster
forecaster = ForecasterRecursiveMultiSeries(
    estimator=RandomForestRegressor(),
    lags=24,
)
forecaster.fit(series=series_data)

# 2. Create adapter
adapter = SkforecastAdapter(forecaster, series=series_data)
X, y = adapter.get_training_data()

# 3. Compute importance (auto-detects groups)
explainer = ConditionalPermutationImportance(
    model=forecaster,
    metric='rmse',
    strategy='auto',
)
result = explainer.explain(X, y, features=['lag_1', 'lag_2'])

# 4. Visualize
fig, ax = plot_importance_bar(result)
```

---

## Type System

Key result types (defined in `timelens.core.types`):

- **`FeatureImportanceResult`**: Container for PFI scores with methods:
  - `to_dataframe()` - Convert to pandas DataFrame
  - Per-feature importance and standard deviation

- **`SHAPResult`**: Container for SHAP values with methods:
  - `mean_abs_shap()` - Mean absolute SHAP per feature
  - Integration with SHAP plotting utilities

---

## Design Patterns

### 1. **Strategy Pattern** (Partitioners)
Different partitioning strategies (auto vs. manual) are interchangeable implementations of `BasePartitioner`, allowing runtime selection.

### 2. **Template Method** (BaseExplainer)
The `BaseExplainer` class provides common initialization and metric resolution, with subclasses implementing `compute()`.

### 3. **Adapter Pattern** (SkforecastAdapter)
Adapts framework-specific models (skforecast) to a common interface for timelens explainers.

### 4. **Composition over Inheritance**
- Explainers use partitioners (composition) rather than inheriting from them
- ConditionalSHAP is standalone, not inheriting from BaseExplainer

---

## Class Diagram

See the mermaid diagrams below for visual representation of inheritance and relationships.
