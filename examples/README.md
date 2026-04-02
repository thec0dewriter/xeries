# Examples

This directory contains Jupyter notebooks demonstrating how to use xeries.

## Notebooks

| Notebook | Description | Dependencies |
|----------|-------------|--------------|
| [01_quickstart.ipynb](01_quickstart.ipynb) | Basic usage with synthetic data | Core xeries |
| [02_skforecast_integration.ipynb](02_skforecast_integration.ipynb) | Integration with skforecast | `xeries[skforecast]` |
| [03_experiment_validation.ipynb](03_experiment_validation.ipynb) | Validation experiments comparing PFI vs cs-PFI methods | `xeries[skforecast]`, `lightgbm` |

## Running the Notebooks

### Install Dependencies

```bash
# Install xeries with notebook support
pip install xeries[notebooks]

# Or with UV
uv add xeries --extra notebooks
```

For skforecast integration:

```bash
pip install xeries[skforecast,notebooks]
```

### Launch Jupyter

```bash
jupyter notebook
```

Then open any notebook from the file browser.

## Testing Notebooks

Notebooks are automatically tested in CI to ensure they execute without errors.

To run notebook tests locally:

```bash
# Structure tests (fast)
pytest tests/test_notebooks.py -m "notebook and not slow"

# Execution tests (slower)
pytest tests/test_notebooks.py -m "notebook and slow"
```
