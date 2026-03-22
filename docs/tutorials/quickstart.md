# Quickstart Tutorial

This tutorial demonstrates a complete workflow using tcpfi with skforecast.

## Setup

First, install the required packages:

```bash
pip install tcpfi[skforecast]
```

## Load Data

We'll use a sample multi-series dataset:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries

# Create sample multi-series data
np.random.seed(42)
n_periods = 500
dates = pd.date_range('2023-01-01', periods=n_periods, freq='h')

data = {}
for series_id in ['store_001', 'store_002', 'store_003']:
    base = np.random.randn() * 10
    trend = np.linspace(0, 5, n_periods)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_periods) / 24)
    noise = np.random.randn(n_periods) * 0.5
    data[series_id] = base + trend + seasonal + noise

series_data = pd.DataFrame(data, index=dates)
print(series_data.head())
```

## Train a Global Model

```python
# Create and train the forecaster
forecaster = ForecasterRecursiveMultiSeries(
    estimator=RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),
    lags=24  # Use 24 hourly lags
)

forecaster.fit(series=series_data)
```

## Extract Training Data

```python
from tcpfi.adapters.skforecast import from_skforecast

# Create adapter (same series as fit — required by skforecast.create_train_X_y)
adapter = from_skforecast(forecaster, series=series_data)

# Get the training matrix
X, y = adapter.get_training_data()
print(f"Training data shape: X={X.shape}, y={y.shape}")
print(f"Features: {adapter.get_feature_names()}")
print(f"Series IDs: {adapter.get_series_ids()}")
```

## Compute Conditional Permutation Importance

### Automatic Strategy (cs-PFI)

```python
from tcpfi import ConditionalPermutationImportance

# Create explainer with automatic tree-based partitioning
explainer = ConditionalPermutationImportance(
    model=adapter,
    metric='mse',
    strategy='auto',
    n_repeats=5,
    random_state=42,
)

# Compute importance for lag features
result = explainer.compute(X, y, features=['lag_1', 'lag_2', 'lag_3'])

# View results
print(result.to_dataframe())
```

### Manual Strategy with Domain Knowledge

```python
from tcpfi import ManualPartitioner

# Domain groups: with wide-format skforecast data, X uses ordinal series codes.
# Keys 0,1,... match forecaster.series_names_in_ column order.
mapping = {
    0: 'region_A',
    1: 'region_B',
    2: 'region_A',
}

partitioner = ManualPartitioner(mapping, series_col=adapter.get_series_column())

explainer_manual = ConditionalPermutationImportance(
    model=adapter,
    metric='mse',
    strategy='manual',
    partitioner=partitioner,
    n_repeats=5,
    random_state=42,
)

result_manual = explainer_manual.compute(X, y, features=['lag_1', 'lag_2'])
print(result_manual.to_dataframe())
```

## Visualize Results

```python
from tcpfi.visualization import plot_importance_bar, plot_importance_heatmap

# Bar plot
fig, ax = plot_importance_bar(result, max_features=10)
fig.savefig('importance_bar.png')

# Compare multiple conditions
results = {
    'Auto (cs-PFI)': result,
    'Manual (Region)': result_manual,
}
fig, ax = plot_importance_heatmap(results)
fig.savefig('importance_comparison.png')
```

## Conditional SHAP

For more detailed explanations:

```python
from tcpfi import ConditionalSHAP

shap_explainer = ConditionalSHAP(
    predict_fn=adapter.predict,
    background_data=X,
    series_col=adapter.get_series_column(),
    n_background_samples=50,
    random_state=42,
)

# Explain a sample of instances
shap_result = shap_explainer.explain(X.iloc[:10])

# Global importance from SHAP
global_shap = shap_explainer.global_importance(X, n_samples=100)
print(global_shap)
```

## Complete Script

Here's the complete example in one script:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries

from tcpfi import ConditionalPermutationImportance, ManualPartitioner
from tcpfi.adapters.skforecast import from_skforecast
from tcpfi.visualization import plot_importance_bar

# Generate data
np.random.seed(42)
n_periods = 500
dates = pd.date_range('2023-01-01', periods=n_periods, freq='h')
series_data = pd.DataFrame({
    f'store_{i:03d}': np.random.randn() * 10 + np.linspace(0, 5, n_periods)
    for i in range(1, 4)
}, index=dates)

# Train forecaster
forecaster = ForecasterRecursiveMultiSeries(
    estimator=RandomForestRegressor(n_estimators=50, random_state=42),
    lags=12
)
forecaster.fit(series=series_data)

# Create adapter and get data
adapter = from_skforecast(forecaster, series=series_data)
X, y = adapter.get_training_data()

# Compute conditional importance
explainer = ConditionalPermutationImportance(
    model=adapter, metric='mse', strategy='auto', n_repeats=3, random_state=42
)
result = explainer.compute(X, y)

# Display and plot
print(result.to_dataframe())
fig, ax = plot_importance_bar(result)
```
