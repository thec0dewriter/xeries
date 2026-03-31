# timelens

[![CI](https://github.com/thec0dewriter/timelens/actions/workflows/ci.yml/badge.svg)](https://github.com/thec0dewriter/timelens/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/timelens.svg)](https://badge.fury.io/py/timelens)
[![Python versions](https://img.shields.io/pypi/pyversions/timelens.svg)](https://pypi.org/project/timelens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Time Series Explainability & Conditional Permutation Feature Importance**

A Python library for generalized time series explainability in multi-time series forecasting, supporting Conditional Feature Importance, SHAP, and future Causal Methods.

## Why timelens?

When using global models for multi-time series forecasting, standard explainability and feature importance methods can produce **misleading results**. They fail to respect the conditional nature of series-dependent features like lags, rolling statistics, and series identifiers.

For example, standard permutation importance might pair a lag value from "Product A" with a target from "Product B" - creating nonsensical data and invalid importance scores.

**timelens** solves this by constraining permutations within meaningful subgroups, preserving the data's structural integrity.

## Features

- **Generic Explainability Interface**: Unified architecture extending beyond permutation importance.
- **Conditional Permutation Importance**: Permute features only within defined subgroups.
- **Tree-Based cs-PFI**: Automatically learn homogeneous subgroups using decision trees.
- **Manual Grouping**: Define custom permutation groups based on domain knowledge.
- **Conditional SHAP & SHAP-IQ**: Series-specific background data for accurate SHAP and interaction explanations.
- **skforecast Integration**: Seamless integration with skforecast's multi-series forecasters.
- **Visualization**: Built-in plotting utilities for importance results.

## Installation

```bash
pip install timelens
```

With UV:

```bash
uv add timelens
```

For skforecast integration:

```bash
pip install timelens[skforecast]
```

## Quick Start

```python
from sklearn.ensemble import RandomForestRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries

from timelens import ConditionalPermutationImportance
from timelens.adapters.skforecast import from_skforecast

# Train your multi-series forecaster (skforecast 0.21+)
forecaster = ForecasterRecursiveMultiSeries(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    lags=24,
)
forecaster.fit(series=your_data)

# Same `series` as fit() is required for create_train_X_y (pass here or to get_training_data)
adapter = from_skforecast(forecaster, series=your_data)
X, y = adapter.get_training_data()

# Compute conditional importance (automatic tree-based cs-PFI)
explainer = ConditionalPermutationImportance(
    model=adapter,
    metric='mse',
    strategy='auto',
    n_repeats=5,
    random_state=42,
)

result = explainer.explain(X, y, features=['lag_1', 'lag_2', 'lag_3'])
print(result.to_dataframe())
```

### Using Manual Groups

```python
from timelens import ManualPartitioner, ConditionalPermutationImportance

# Domain groups: with skforecast 0.21+ wide data, series are ordinal-encoded in X.
# Map integers 0,1,... in the same order as forecaster.series_names_in_ (see adapter.forecaster).
mapping = {
    0: 'urban',
    1: 'suburban',
    2: 'urban',
}

partitioner = ManualPartitioner(mapping, series_col=adapter.get_series_column())

explainer = ConditionalPermutationImportance(
    model=adapter,
    metric='mse',
    strategy='manual',
    partitioner=partitioner,
)

result = explainer.explain(X, y)
```

### Conditional SHAP

```python
from timelens import ConditionalSHAP

explainer = ConditionalSHAP(
    model=adapter,
    background_data=X,
    # skforecast 0.21+: use adapter.get_series_column() (often "_level_skforecast")
    series_col=adapter.get_series_column(),
    n_background_samples=100,
)

# Explain instances with series-specific backgrounds
shap_result = explainer.explain(X.iloc[:10])

# Global importance
global_importance = explainer.global_importance(X, n_samples=100)
```

### Visualization

```python
from timelens.visualization import plot_importance_bar, plot_importance_heatmap

# Bar chart
fig, ax = plot_importance_bar(result, max_features=10)

# Heatmap comparing multiple results
results = {'Auto': result_auto, 'Manual': result_manual}
fig, ax = plot_importance_heatmap(results)
```

## Documentation

Full documentation is available at [https://thec0dewriter.github.io/timelens](https://thec0dewriter.github.io/timelens)

## Development

Clone the repository:

```bash
git clone https://github.com/thec0dewriter/timelens.git
cd timelens
```

Install with development dependencies:

```bash
uv sync --dev
```

Run tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check src tests
uv run ruff format src tests
```

Type checking:

```bash
uv run ty check src
```

Build documentation:

```bash
uv sync --group docs
uv run mkdocs serve
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{timelens,
  title = {timelens: Time-Conditional Permutation Feature Importance},
  author = {thec0dewriter},
  year = {2026},
  url = {https://github.com/thec0dewriter/timelens}
}
```

## Acknowledgments

This library implements methods inspired by:

- Conditional Subgroup Permutation Feature Importance (cs-PFI)
- SHAP (SHapley Additive exPlanations)
- skforecast multi-series forecasting framework
