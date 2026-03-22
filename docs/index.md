# tcpfi

**Time-Conditional Permutation Feature Importance**

A Python library for conditional feature importance in multi-time series forecasting.

## Overview

When using global models for multi-time series forecasting, standard feature importance methods can produce misleading results because they fail to respect the conditional nature of series-dependent features (like lags and series identifiers). This leads to unrealistic data permutations and unreliable insights.

**tcpfi** addresses this challenge by providing:

- **Conditional Permutation Importance**: Permutes features only within meaningful subgroups
- **Tree-Based cs-PFI**: Automatically learns homogeneous subgroups using decision trees
- **Manual Grouping**: Use domain knowledge to define custom permutation groups
- **Conditional SHAP**: Series-specific background data for accurate SHAP values
- **Framework Integration**: Works with skforecast, and extensible to other frameworks

## Installation

```bash
pip install tcpfi
```

Or with UV:

```bash
uv add tcpfi
```

For skforecast integration:

```bash
pip install tcpfi[skforecast]
```

## Quick Example

```python
import tcpfi
from sklearn.ensemble import RandomForestRegressor

# Assume you have a trained model and data
explainer = tcpfi.ConditionalPermutationImportance(
    model=model,
    metric='mse',
    strategy='auto'  # Uses tree-based cs-PFI
)

result = explainer.compute(X, y, features=['lag_1', 'lag_2', 'day_of_week'])
print(result.to_dataframe())
```

## Key Concepts

### Series-Dependent Features

In multi-series forecasting, many features are intrinsically tied to their series:

- **Autoregressive lags**: `lag_1` for Product A is meaningless for Product B
- **Rolling statistics**: A 7-day rolling mean is series-specific
- **Series identifiers**: Explicitly tell the model which series a row belongs to

### Why Conditional Importance?

Standard permutation importance shuffles feature values across the entire dataset, potentially pairing a lag from "Product A" with a target from "Product B". This creates nonsensical data and invalid importance scores.

**Conditional importance** constrains permutations within subgroups where the permutation makes sense, preserving the data's structural integrity.

## Next Steps

- [Getting Started](getting-started.md): Detailed setup and first steps
- [Quickstart Tutorial](tutorials/quickstart.md): End-to-end example with skforecast
- [API Reference](api/reference.md): Complete API documentation
