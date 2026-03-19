# tcpfi

[![CI](https://github.com/thec0dewriter/time_conditional_pfi/actions/workflows/ci.yml/badge.svg)](https://github.com/thec0dewriter/time_conditional_pfi/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/tcpfi.svg)](https://badge.fury.io/py/tcpfi)
[![Python versions](https://img.shields.io/pypi/pyversions/tcpfi.svg)](https://pypi.org/project/tcpfi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Time-Conditional Permutation Feature Importance**

A Python library for conditional feature importance in multi-time series forecasting.

## Why tcpfi?

When using global models for multi-time series forecasting, standard feature importance methods can produce **misleading results**. They fail to respect the conditional nature of series-dependent features like lags, rolling statistics, and series identifiers.

For example, standard permutation importance might pair a lag value from "Product A" with a target from "Product B" - creating nonsensical data and invalid importance scores.

**tcpfi** solves this by constraining permutations within meaningful subgroups, preserving the data's structural integrity.

## Features

- **Conditional Permutation Importance**: Permute features only within defined subgroups
- **Tree-Based cs-PFI**: Automatically learn homogeneous subgroups using decision trees
- **Manual Grouping**: Define custom permutation groups based on domain knowledge
- **Conditional SHAP**: Series-specific background data for accurate SHAP explanations
- **skforecast Integration**: Seamless integration with skforecast's multi-series forecasters
- **Visualization**: Built-in plotting utilities for importance results

## Installation

```bash
pip install tcpfi
```

With UV:

```bash
uv add tcpfi
```

For skforecast integration:

```bash
pip install tcpfi[skforecast]
```

## Quick Start

```python
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterMultiSeries import ForecasterMultiSeries

from tcpfi import ConditionalPermutationImportance
from tcpfi.adapters.skforecast import from_skforecast

# Train your multi-series forecaster
forecaster = ForecasterMultiSeries(
    regressor=RandomForestRegressor(n_estimators=100, random_state=42),
    lags=24
)
forecaster.fit(series=your_data)

# Create adapter and get training data
adapter = from_skforecast(forecaster)
X, y = adapter.get_training_data()

# Compute conditional importance (automatic tree-based cs-PFI)
explainer = ConditionalPermutationImportance(
    model=adapter,
    metric='mse',
    strategy='auto',
    n_repeats=5,
    random_state=42,
)

result = explainer.compute(X, y, features=['lag_1', 'lag_2', 'lag_3'])
print(result.to_dataframe())
```

### Using Manual Groups

```python
from tcpfi import ManualPartitioner, ConditionalPermutationImportance

# Define groups based on domain knowledge
mapping = {
    'store_001': 'urban',
    'store_002': 'suburban',
    'store_003': 'urban',
}

partitioner = ManualPartitioner(mapping, series_col='level')

explainer = ConditionalPermutationImportance(
    model=adapter,
    metric='mse',
    strategy='manual',
    partitioner=partitioner,
)

result = explainer.compute(X, y)
```

### Conditional SHAP

```python
from tcpfi import ConditionalSHAP

explainer = ConditionalSHAP(
    predict_fn=adapter.predict,
    background_data=X,
    series_col='level',
    n_background_samples=100,
)

# Explain instances with series-specific backgrounds
shap_result = explainer.explain(X.iloc[:10])

# Global importance
global_importance = explainer.global_importance(X, n_samples=100)
```

### Visualization

```python
from tcpfi.visualization import plot_importance_bar, plot_importance_heatmap

# Bar chart
fig, ax = plot_importance_bar(result, max_features=10)

# Heatmap comparing multiple results
results = {'Auto': result_auto, 'Manual': result_manual}
fig, ax = plot_importance_heatmap(results)
```

## Documentation

Full documentation is available at [https://thec0dewriter.github.io/time_conditional_pfi](https://thec0dewriter.github.io/time_conditional_pfi)

## Development

Clone the repository:

```bash
git clone https://github.com/thec0dewriter/time_conditional_pfi.git
cd time_conditional_pfi
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
uv run mypy src
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
@software{tcpfi,
  title = {tcpfi: Time-Conditional Permutation Feature Importance},
  author = {thec0dewriter},
  year = {2026},
  url = {https://github.com/thec0dewriter/time_conditional_pfi}
}
```

## Acknowledgments

This library implements methods inspired by:

- Conditional Subgroup Permutation Feature Importance (cs-PFI)
- SHAP (SHapley Additive exPlanations)
- skforecast multi-series forecasting framework
