# xeries

[![CI](https://github.com/thec0dewriter/xeries/actions/workflows/ci.yml/badge.svg)](https://github.com/thec0dewriter/xeries/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/xeries.svg)](https://badge.fury.io/py/xeries)
[![Python versions](https://img.shields.io/pypi/pyversions/xeries.svg)](https://pypi.org/project/xeries/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Time Series eXplainability (XAI) for Forecasting**

A comprehensive Python library for explainability and interpretability in multi-time series forecasting. **xeries** provides multiple explanation methods—including conditional permutation importance, SHAP, feature dropping, and causal analysis—with a unified API and interactive dashboard for understanding forecast decisions.

## Why xeries?

Explaining and interpreting multi-time series forecasts is challenging:

1. **Standard methods fail on series-dependent features**: Permutation importance and SHAP can create invalid data by pairing a lag from one series with a target from another.

2. **No unified explainability framework**: Most libraries offer a single method (e.g., SHAP only) with limited support for time-series-specific analysis.

3. **Lack of temporal and comparative insights**: Understanding performance across time windows, detecting significance, and comparing methods are difficult without custom tools.

**xeries** addresses these gaps with:
- **Conditional explanations** that respect data structure
- **Multiple methods** so you can triangulate insights
- **Time-aware analytics** for temporal patterns
- **Unified API** for seamless method switching
- **Interactive dashboard** for exploratory analysis

## Features

### 📊 Explainability Methods

- **Conditional Permutation Importance (cs-PFI)**: Permute features only within meaningful subgroups
  - Auto-discover groups using decision trees
  - Define custom groups based on domain knowledge
- **Conditional SHAP**: Compute SHAP values with series-specific background data
- **SHAP-IQ**: Analyze feature interactions and higher-order effects
- **Feature Dropping**: Measure importance by removing features
- **Causal Feature Importance**: Causal inference for treatment effects (DoWhy + EconML integration)

### 🛠️ Advanced Analytics

- **Temporal Windowed Analysis**: Importance decomposed across time windows
- **Statistical Significance Testing**: Bootstrap confidence intervals and hypothesis tests
- **Method Comparison**: Side-by-side results from multiple explanation methods
- **Error Analysis**: Per-series and per-window error metrics and attribution
- **Feature Interaction Analysis**: Understand how features work together

### 📡 Framework Adapters

- **scikit-learn**: Direct support for sklearn estimators
- **skforecast**: Seamless integration with multi-series forecasters (0.21+)
- **Darts (PyTorch)**: Support for Darts neural network forecasters
- **Custom Models**: Wrap any forecaster with the BaseAdapter

### 📈 Visualization & Reporting

- **Interactive Dashboard**: Unified interface for all explainability components
- **Publication-Ready Plots**:
  - Feature importance bar charts
  - Temporal heatmaps
  - Method comparison visualizations
  - Interaction plots
- **HTML Report Generation**: Auto-generate dashboards with Jinja2 templates
- **Jupyter Integration**: Works seamlessly in notebooks

## Installation

```bash
pip install xeries
```

With UV:

```bash
uv add xeries
```

For skforecast integration:

```bash
pip install xeries[skforecast]
```

## Supported Explanation Methods

| Method | Type | Use Case | Features |
|--------|------|----------|----------|
| **Conditional Permutation Importance** | Model-Agnostic | Default choice; fast & interpretable | Auto/manual grouping, windowed analysis |
| **Conditional SHAP** | Additive | Local & global explanations | Series-aware backgrounds, force plots |
| **SHAP-IQ** | Interaction | Feature interactions | Shapley interaction values, comparative |
| **Feature Dropping** | Model-Agnostic | Complementary to importance | Dependency analysis, isolation effects |
| **Causal Feature Importance** | Causal | Treatment effects | DoWhy pipelines, EconML estimators |

## Quick Start

### Conditional Permutation Importance (Default)

```python
from sklearn.ensemble import RandomForestRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries

from xeries import ConditionalPermutationImportance
from xeries.adapters.skforecast import from_skforecast

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
from xeries import ManualPartitioner, ConditionalPermutationImportance

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
from xeries import ConditionalSHAP

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
from xeries.visualization import plot_importance_bar, plot_importance_heatmap

# Bar chart
fig, ax = plot_importance_bar(result, max_features=10)

# Heatmap comparing multiple results
results = {'Auto': result_auto, 'Manual': result_manual}
fig, ax = plot_importance_heatmap(results)
```

### Causal Feature Importance

```python
from xeries import CausalFeatureImportance

# Analyze causal treatment effects with DoWhy backend
explainer = CausalFeatureImportance(
    model=adapter,
    treatment_features=['lag_1', 'lag_24'],  # Which features to treat
)

result = explainer.explain(X, y)
print(result.estimates)  # Causal effect estimates
print(result.refutations)  # Robustness checks
```

### Compare Multiple Methods

```python
from xeries import (
    ConditionalPermutationImportance,
    ConditionalSHAP,
    ConditionalDropImportance,
)
from xeries.analysis import compare_rankings

# Compute explanations with different methods
pfi_result = ConditionalPermutationImportance(...).explain(X, y)
shap_result = ConditionalSHAP(...).explain(X)
drop_result = ConditionalDropImportance(...).explain(X, y)

# Compare results
comparison = compare_rankings(
    {'PFI': pfi_result, 'SHAP': shap_result, 'Dropping': drop_result}
)
```

### Unified Dashboard

```python
from xeries import Dashboard

# Combine all explainability results into one interactive dashboard
dashboard = Dashboard(forecaster=adapter)
dashboard.add_permutation_importance(pfi_result)
dashboard.add_causal_importance(causal_result)
dashboard.add_error_analysis(X, y, predictions)

# Generate HTML report
dashboard.generate_report('forecast_analysis.html')

# Or display in Jupyter
dashboard.show()
```

## Documentation

Full documentation is available at [https://thec0dewriter.github.io/xeries](https://thec0dewriter.github.io/xeries)

## Development

Clone the repository:

```bash
git clone https://github.com/thec0dewriter/xeries.git
cd xeries
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
@software{xeries,
  title = {xeries: Time Series eXplainability for Forecasting},
  author = {thec0dewriter},
  year = {2026},
  url = {https://github.com/thec0dewriter/xeries},
}
```

### Related Publications

This library implements techniques from:

- **Conditional Subgroup Permutation Feature Importance** (cs-PFI)
- **SHAP** (SHapley Additive exPlanations) — [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
- **SHAP Interaction Values** — [Janzing et al., 2020](https://arxiv.org/abs/1908.08474)
- **Controlled Feature Dropping** — Model modification for importance estimation
- **Causal Inference** using **DoWhy** and **EconML** frameworks
- **skforecast** — [Multi-series forecasting framework](https://skforecast.org/)
