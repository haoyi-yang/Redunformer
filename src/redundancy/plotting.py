"""Plotting helpers for redundancy visualizations (Weeks 5–8 measurement)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _prep(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_heatmap(matrix, path: str | Path, title: str = "", xlabel: str = "", ylabel: str = "") -> Path:
    path = _prep(path)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_frequency_histograms(
    freqs_by_layer: dict[int, np.ndarray],
    path: str | Path,
    *,
    bins: int = 40,
    title: str = "Per-neuron activation frequency by layer",
    max_cols: int = 4,
) -> Path:
    """Small-multiple histograms of activation frequency, one panel per layer."""
    path = _prep(path)
    layers = sorted(freqs_by_layer)
    n = len(layers)
    cols = min(max_cols, n) or 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), squeeze=False)
    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx // cols][ax_idx % cols]
        ax.hist(freqs_by_layer[layer], bins=bins, range=(0.0, 1.0), color="#4c72b0")
        ax.set_title(f"layer {layer}", fontsize=9)
        ax.set_yscale("log")
        ax.tick_params(labelsize=7)
    for empty in range(n, rows * cols):
        axes[empty // cols][empty % cols].axis("off")
    fig.suptitle(title)
    fig.supxlabel("activation frequency f")
    fig.supylabel("neuron count (log)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_freq_distribution_heatmap(
    freqs_by_layer: dict[int, np.ndarray],
    path: str | Path,
    *,
    bins: int = 40,
    title: str = "Activation-frequency distribution across depth",
) -> Path:
    """Heatmap: rows = layers, columns = frequency bins, color = neuron fraction."""
    path = _prep(path)
    layers = sorted(freqs_by_layer)
    edges = np.linspace(0.0, 1.0, bins + 1)
    grid = np.zeros((len(layers), bins))
    for r, layer in enumerate(layers):
        hist, _ = np.histogram(freqs_by_layer[layer], bins=edges)
        total = hist.sum() or 1
        grid[r] = hist / total
    plt.figure(figsize=(9, max(4, 0.25 * len(layers))))
    ax = sns.heatmap(
        grid,
        cmap="magma",
        cbar_kws={"label": "neuron fraction"},
        norm=None,
    )
    xticks = np.linspace(0, bins, 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 6)])
    ax.set_yticks(np.arange(len(layers)) + 0.5)
    ax.set_yticklabels(layers, fontsize=7)
    ax.set_xlabel("activation frequency f")
    ax.set_ylabel("layer index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_lazy_fraction_by_depth(
    layer_indices: Sequence[int],
    lazy_fraction: Sequence[float],
    path: str | Path,
    *,
    title: str = "Lazy-neuron fraction vs depth",
    ylabel: str = "fraction of neurons with f \u2264 1%",
) -> Path:
    """Bar chart of the lazy-neuron fraction per layer, with depth-tertile shading."""
    path = _prep(path)
    order = np.argsort(layer_indices)
    xs = np.asarray(layer_indices)[order]
    ys = np.asarray(lazy_fraction)[order]
    plt.figure(figsize=(10, 4))
    plt.bar(xs, ys, color="#dd8452", width=0.8)
    n = len(xs)
    if n >= 3:
        third = n / 3.0
        plt.axvspan(xs[0] - 0.5, xs[int(third) - 1] + 0.5, color="#55a868", alpha=0.08)
        plt.axvspan(xs[int(2 * third)] - 0.5, xs[-1] + 0.5, color="#c44e52", alpha=0.08)
    plt.xlabel("layer index (early \u2192 deep)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_metric_by_depth(
    layer_indices: Sequence[int],
    values: Sequence[float],
    path: str | Path,
    *,
    title: str,
    ylabel: str,
    color: str = "#4c72b0",
) -> Path:
    """Generic per-layer bar chart with early/deep tertile shading."""
    path = _prep(path)
    order = np.argsort(layer_indices)
    xs = np.asarray(layer_indices)[order]
    ys = np.asarray(values, dtype=float)[order]
    plt.figure(figsize=(10, 4))
    plt.bar(xs, ys, color=color, width=0.8)
    n = len(xs)
    if n >= 3:
        third = n / 3.0
        plt.axvspan(xs[0] - 0.5, xs[int(third) - 1] + 0.5, color="#55a868", alpha=0.08)
        plt.axvspan(xs[int(2 * third)] - 0.5, xs[-1] + 0.5, color="#c44e52", alpha=0.08)
    plt.xlabel("layer index (early \u2192 deep)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_importance_histograms(
    importance_by_layer: dict[int, np.ndarray],
    path: str | Path,
    *,
    bins: int = 40,
    title: str = "Per-neuron importance by layer (RMS x weight-col-norm)",
    max_cols: int = 4,
) -> Path:
    """Small-multiple log-log histograms of neuron importance, one per layer."""
    path = _prep(path)
    layers = sorted(importance_by_layer)
    n = len(layers)
    cols = min(max_cols, n) or 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), squeeze=False)
    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx // cols][ax_idx % cols]
        vals = np.asarray(importance_by_layer[layer], dtype=float)
        vals = vals[vals > 0]
        if vals.size:
            log_bins = np.logspace(np.log10(vals.min() + 1e-12), np.log10(vals.max() + 1e-12), bins)
            ax.hist(vals, bins=log_bins, color="#937860")
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"layer {layer}", fontsize=9)
        ax.tick_params(labelsize=7)
    for empty in range(n, rows * cols):
        axes[empty // cols][empty % cols].axis("off")
    fig.suptitle(title)
    fig.supxlabel("neuron importance (log)")
    fig.supylabel("neuron count (log)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_lazy_fraction_sweep(
    layer_indices: Sequence[int],
    fraction_by_alpha: dict[float, Sequence[float]],
    path: str | Path,
    *,
    title: str = "Lazy-neuron fraction vs depth (threshold sweep)",
) -> Path:
    """One line per RMS-relative threshold alpha: lazy fraction across depth."""
    path = _prep(path)
    order = np.argsort(layer_indices)
    xs = np.asarray(layer_indices)[order]
    plt.figure(figsize=(10, 4.5))
    cmap = plt.get_cmap("viridis")
    alphas = sorted(fraction_by_alpha)
    for idx, alpha in enumerate(alphas):
        ys = np.asarray(fraction_by_alpha[alpha], dtype=float)[order]
        color = cmap(idx / max(len(alphas) - 1, 1))
        plt.plot(xs, ys, marker="o", ms=3, lw=1.4, color=color, label=f"\u03b1={alpha:g}")
    plt.xlabel("layer index (early \u2192 deep)")
    plt.ylabel("fraction of neurons firing on \u2264 1% of tokens")
    plt.title(title)
    plt.legend(title="threshold = \u03b1\u00b7RMS", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_removal_curves(
    ratios: Sequence[float],
    series: dict[str, Sequence[float]],
    path: str | Path,
    *,
    title: str = "Performance vs neuron removal ratio",
    ylabel: str = "WikiText-2 perplexity",
    baseline: float | None = None,
    logy: bool = False,
) -> Path:
    """Performance-vs-removal curve, one line per ranking strategy.

    The deliverable is the *gap* between the guided curves and the random
    baseline at matched sparsity, so all strategies share one axis.
    """
    path = _prep(path)
    xs = np.asarray(ratios, dtype=float) * 100.0
    plt.figure(figsize=(8, 5))
    palette = {"random": "#8c8c8c"}
    cmap = plt.get_cmap("tab10")
    for idx, (label, values) in enumerate(sorted(series.items())):
        color = palette.get(label, cmap(idx % 10))
        style = "--" if label == "random" else "-"
        plt.plot(
            xs,
            np.asarray(values, dtype=float),
            marker="o",
            ms=4,
            lw=1.6,
            ls=style,
            color=color,
            label=label,
        )
    if baseline is not None:
        plt.axhline(baseline, color="#333333", lw=1.0, ls=":", label="unmasked baseline")
    if logy:
        plt.yscale("log")
    plt.xlabel("neurons masked per layer (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_family_removal_curves(
    ratios: Sequence[float],
    series: dict[str, dict[str, Sequence[float]]],
    path: str | Path,
    *,
    title: str = "Neuron removal cost across model families",
    ylabel: str = "perplexity relative to unmasked baseline",
    logy: bool = False,
) -> Path:
    """Overlay removal curves for several models, colour by model, style by strategy.

    ``series`` maps a model label to ``{strategy: values}``. Absolute perplexity
    is not comparable across families, so callers should pass a ratio to each
    model's own baseline; the readable signal is then how far the guided curve
    sits below the random one for each family.
    """
    path = _prep(path)
    xs = np.asarray(ratios, dtype=float) * 100.0
    styles = {"importance": "-", "frequency": "-.", "random": "--"}
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(9, 5.5))
    for idx, (model, strategies) in enumerate(series.items()):
        color = cmap(idx % 10)
        for strategy, values in sorted(strategies.items()):
            plt.plot(
                xs,
                np.asarray(values, dtype=float),
                marker="o" if strategy == "importance" else None,
                ms=4,
                lw=1.7 if strategy == "importance" else 1.2,
                ls=styles.get(strategy, ":"),
                color=color,
                alpha=1.0 if strategy == "importance" else 0.65,
                label=f"{model} / {strategy}",
            )
    plt.axhline(1.0, color="#333333", lw=1.0, ls=":", label="unmasked baseline")
    if logy:
        plt.yscale("log")
    plt.xlabel("neurons masked per layer (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=7, ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_condition_bars(
    labels: Sequence[str],
    values: Sequence[float],
    path: str | Path,
    *,
    title: str,
    ylabel: str,
    highlight: Sequence[str] = (),
) -> Path:
    """Bar chart comparing ablation conditions (used for the H5 pair study)."""
    path = _prep(path)
    ys = np.asarray(values, dtype=float)
    colors = ["#c44e52" if label in set(highlight) else "#4c72b0" for label in labels]
    plt.figure(figsize=(max(6, 1.4 * len(labels)), 4.5))
    bars = plt.bar(range(len(labels)), ys, color=colors, width=0.65)
    span = float(np.max(np.abs(ys))) if ys.size else 0.0
    for bar, value in zip(bars, ys):
        offset = 0.02 * span if span else 0.01
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + (offset if value >= 0 else -offset),
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    plt.axhline(0.0, color="#333333", lw=0.8)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_overlap_by_depth(
    layer_indices: Sequence[int],
    series: dict[str, Sequence[float]],
    path: str | Path,
    *,
    title: str = "Cross-task neuron agreement by depth",
    ylabel: str = "Jaccard overlap",
    chance: float | None = None,
) -> Path:
    """Per-layer set agreement across depth, with the random-agreement floor.

    The ``chance`` line is what matters for reading these curves: a Jaccard of
    0.05 is meaningless on its own but is the expected value for two independent
    10% selections.
    """
    path = _prep(path)
    xs = np.asarray(layer_indices, dtype=int)
    plt.figure(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for idx, (label, values) in enumerate(sorted(series.items())):
        plt.plot(
            xs,
            np.asarray(values, dtype=float),
            marker="o",
            ms=4,
            lw=1.6,
            color=cmap(idx % 10),
            label=label,
        )
    if chance is not None:
        plt.axhline(chance, color="#8c8c8c", lw=1.0, ls=":", label=f"chance ({chance:.3f})")
    plt.ylim(bottom=0.0)
    plt.xlabel("layer index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    path: str | Path,
    *,
    title: str = "Neuron correlation (selected layer)",
) -> Path:
    """Heatmap of a (sub)matrix of neuron correlations, centered at 0."""
    path = _prep(path)
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0.0, vmin=-1.0, vmax=1.0)
    plt.title(title)
    plt.xlabel("neuron")
    plt.ylabel("neuron")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
