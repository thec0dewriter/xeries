"""Visualization utilities for feature importance results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from tcpfi.core.types import FeatureImportanceResult, SHAPResult


def plot_importance_bar(
    result: FeatureImportanceResult,
    max_features: int | None = 20,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str | None = None,
    color: str = "#1f77b4",
    show_std: bool = True,
) -> tuple[Figure, Axes]:
    """Plot feature importance as a horizontal bar chart.

    Args:
        result: FeatureImportanceResult from an explainer.
        max_features: Maximum number of features to display (top N).
        ax: Matplotlib axes to plot on. If None, creates new figure.
        figsize: Figure size (width, height) in inches.
        title: Plot title. If None, uses default.
        color: Bar color.
        show_std: Whether to show error bars for standard deviation.

    Returns:
        Tuple of (Figure, Axes).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    df = result.to_dataframe()
    if max_features is not None:
        df = df.head(max_features)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast("Figure", ax.get_figure())

    y_pos = np.arange(len(df))
    importances = df["importance"].values

    xerr = df["std"].values if show_std and "std" in df.columns else None

    ax.barh(y_pos, importances, xerr=xerr, color=color, alpha=0.8, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (increase in error)")
    ax.set_title(title or "Conditional Permutation Feature Importance")

    plt.tight_layout()
    return fig, ax


def plot_importance_heatmap(
    results: dict[str, FeatureImportanceResult],
    features: list[str] | None = None,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (12, 8),
    cmap: str = "YlOrRd",
    annot: bool = True,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot importance comparison across multiple conditions as a heatmap.

    Args:
        results: Dictionary mapping condition names to FeatureImportanceResult.
        features: List of features to include. If None, uses union of all.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        figsize: Figure size (width, height) in inches.
        cmap: Colormap name.
        annot: Whether to annotate cells with values.
        title: Plot title.

    Returns:
        Tuple of (Figure, Axes).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    data_dict: dict[str, dict[str, float]] = {}
    for condition, result in results.items():
        data_dict[condition] = dict(zip(result.feature_names, result.importances, strict=True))

    df = pd.DataFrame(data_dict)

    if features is not None:
        df = df.loc[df.index.isin(features)]

    df = df.sort_values(by=list(df.columns), ascending=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast("Figure", ax.get_figure())

    im = ax.imshow(df.values, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if annot:
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                value = df.iloc[i, j]
                text_color = "white" if value > df.values.max() * 0.5 else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color)

    fig.colorbar(im, ax=ax, label="Importance")
    ax.set_title(title or "Feature Importance Comparison")

    plt.tight_layout()
    return fig, ax


def plot_shap_summary(
    result: SHAPResult,
    max_features: int | None = 20,
    figsize: tuple[int, int] = (10, 8),
    title: str | None = None,
) -> tuple[Figure, Any]:
    """Plot SHAP summary using the shap library's built-in visualization.

    Args:
        result: SHAPResult from ConditionalSHAP.
        max_features: Maximum number of features to display.
        figsize: Figure size (width, height) in inches.
        title: Plot title.

    Returns:
        Tuple of (Figure, Axes).
    """
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as e:
        raise ImportError(
            "shap and matplotlib are required for plotting. "
            "Install with: pip install shap matplotlib"
        ) from e

    fig, ax = plt.subplots(figsize=figsize)

    shap.summary_plot(
        result.shap_values,
        result.data,
        feature_names=result.feature_names,
        max_display=max_features or len(result.feature_names),
        show=False,
    )

    if title:
        plt.title(title)

    return fig, ax


def plot_shap_bar(
    result: SHAPResult,
    max_features: int | None = 20,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str | None = None,
    color: str = "#ff7f0e",
) -> tuple[Figure, Axes]:
    """Plot mean absolute SHAP values as a bar chart.

    Args:
        result: SHAPResult from ConditionalSHAP.
        max_features: Maximum number of features to display.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        figsize: Figure size (width, height) in inches.
        title: Plot title.
        color: Bar color.

    Returns:
        Tuple of (Figure, Axes).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    df: pd.DataFrame = result.mean_abs_shap()
    if max_features is not None:
        df = df.head(max_features)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast("Figure", ax.get_figure())

    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["mean_abs_shap"].values, color=color, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title or "Conditional SHAP Feature Importance")

    plt.tight_layout()
    return fig, ax
