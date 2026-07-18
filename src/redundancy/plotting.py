"""Plotting helpers for redundancy measurement results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for scripts/CI

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_bi_bar(bi_scores: np.ndarray, out_path: str, title: str = "Block Influence per layer") -> Path:
    """Bar chart of the per-layer Block Influence score. Low BI = redundant block."""
    fig, ax = plt.subplots(figsize=(8, 4))
    layers = np.arange(len(bi_scores))
    ax.bar(layers, bi_scores, color="steelblue")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Block Influence (BI)")
    ax.set_title(title)
    ax.set_xticks(layers)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_similarity_heatmap(matrix: np.ndarray, out_path: str, title: str = "Layer similarity") -> Path:
    """Heatmap of an (L+1)x(L+1) similarity matrix across all layer pairs."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, vmin=-1.0, vmax=1.0, cmap="viridis", square=True, ax=ax)
    # Heatmap is indexed by hidden state (0..L), NOT block (0..L-1): hidden state 0 is the
    # embedding output, hidden state L is the final block's output. BI block i is the i->i+1
    # transition, so it does not line up 1:1 with a single heatmap index.
    ax.set_xlabel("Hidden state (0 = embeddings, 12 = final)")
    ax.set_ylabel("Hidden state (0 = embeddings, 12 = final)")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_residual_norms(residual_norms: np.ndarray, out_path: str, title: str = "Relative residual update per block") -> Path:
    """Bar chart of mean ||X_{i+1} - X_i|| / ||X_i|| per block, as a cross-check on the BI score."""
    fig, ax = plt.subplots(figsize=(8, 4))
    layers = np.arange(len(residual_norms))
    ax.bar(layers, residual_norms, color="darkorange")
    # Log scale: block 0's update (~10.9, inflated by the small embedding-norm denominator)
    # otherwise squashes blocks 1-11 into a flat strip and hides the block-11 elevation that
    # corroborates the BI ranking.
    ax.set_yscale("log")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Mean relative residual update (log scale)")
    ax.set_title(title)
    ax.set_xticks(layers)
    fig.tight_layout()
    return _save(fig, out_path)


def _save(fig, out_path: str) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure to {path}")
    return path
