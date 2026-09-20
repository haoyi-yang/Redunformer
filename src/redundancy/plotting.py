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


def plot_similarity_heatmap(
    matrix: np.ndarray,
    out_path: str,
    title: str = "Layer similarity",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> Path:
    """Heatmap of an (L+1)x(L+1) similarity matrix across all layer pairs."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, vmin=vmin, vmax=vmax, cmap="viridis", square=True, ax=ax)
    # Heatmap is indexed by hidden state (0..L), NOT block (0..L-1): hidden state 0 is the
    # embedding output, hidden state L is the final block's output. BI block i is the i->i+1
    # transition, so it does not line up 1:1 with a single heatmap index.
    final_index = matrix.shape[0] - 1
    axis_label = f"Hidden state (0 = embeddings, {final_index} = final)"
    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)
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


def plot_lm_eval_curve(
    acc_norm: list[float],
    acc_stderr: list[float],
    blocks_removed: list[int] | None = None,
    out_path: str = "reports/group9/figures/lm_eval.png",
    title: str = "Pruning Evaluation — HellaSwag Accuracy",
    baseline_acc: float | None = None,
    baseline_stderr: float | None = None,
    perplexity: list[float] | None = None,
    baseline_ppl: float | None = None,
) -> Path:
    """Plot lm_eval normalized accuracy (and optionally perplexity) across pruning steps.

    Args:
        acc_norm: List of acc_norm values for k=1..N removed blocks.
        acc_stderr: List of standard errors for acc_norm.
        blocks_removed: Sequence of block indices removed at each step.
        out_path: Destination image path.
        title: Plot super-title or title.
        baseline_acc: Accuracy of the unpruned model (k=0).
        baseline_stderr: Standard error of unpruned model accuracy.
        perplexity: Optional list of perplexity values for k=1..N.
        baseline_ppl: Optional perplexity of the unpruned model.
    """
    has_ppl = perplexity is not None and len(perplexity) > 0

    if has_ppl:
        fig, (ax_acc, ax_ppl) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, ax_acc = plt.subplots(figsize=(8, 5))

    # Prepare k and accuracy data
    k_vals = list(range(1, len(acc_norm) + 1))
    plot_k = list(k_vals)
    plot_acc = list(acc_norm)
    plot_err = list(acc_stderr)

    if baseline_acc is not None:
        plot_k = [0] + plot_k
        plot_acc = [baseline_acc] + plot_acc
        base_err = baseline_stderr if baseline_stderr is not None else 0.0
        plot_err = [base_err] + plot_err

    # Plot Accuracy
    ax_acc.errorbar(
        plot_k,
        plot_acc,
        yerr=plot_err,
        fmt="-o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        markersize=7,
        linewidth=2,
        label="HellaSwag (acc_norm)",
    )

    if baseline_acc is not None:
        ax_acc.axhline(
            baseline_acc,
            color="#2ca02c",
            linestyle="--",
            alpha=0.7,
            label=f"Unpruned baseline ({baseline_acc:.4f})",
        )

    # 25% random guessing baseline for 4-choice tasks
    ax_acc.axhline(
        0.25,
        color="#d62728",
        linestyle=":",
        alpha=0.6,
        label="Random chance (0.25)",
    )

    # Annotate points with block removed
    if blocks_removed:
        for idx, blk in enumerate(blocks_removed):
            k = idx + 1
            if k in plot_k:
                pos = plot_k.index(k)
                y = plot_acc[pos]
                offset_y = 10 if idx % 2 == 0 else -16
                ax_acc.annotate(
                    f"-B{blk}",
                    xy=(k, y),
                    xytext=(0, offset_y),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    fontweight="semibold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="#1f77b4", linewidth=0.8),
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8),
                )

    ax_acc.set_xlabel("Number of blocks removed (k)", fontsize=11)
    ax_acc.set_ylabel("Accuracy (Normalized)", fontsize=11)
    ax_acc.set_title("HellaSwag Normalized Accuracy", fontsize=12, pad=10)
    ax_acc.set_xticks(plot_k)
    ax_acc.grid(True, linestyle="--", alpha=0.4)
    ax_acc.legend(loc="best", framealpha=0.9)

    # Plot Perplexity if requested
    if has_ppl:
        ppl_k = list(k_vals)
        plot_ppl = list(perplexity)
        if baseline_ppl is not None:
            ppl_k = [0] + ppl_k
            plot_ppl = [baseline_ppl] + plot_ppl

        ax_ppl.plot(
            ppl_k,
            plot_ppl,
            "-s",
            color="#ff7f0e",
            linewidth=2,
            markersize=7,
            label="WikiText-2 Perplexity",
        )
        if baseline_ppl is not None:
            ax_ppl.axhline(
                baseline_ppl,
                color="#2ca02c",
                linestyle="--",
                alpha=0.7,
                label=f"Unpruned baseline ({baseline_ppl:.2f})",
            )

        if blocks_removed:
            for idx, blk in enumerate(blocks_removed):
                k = idx + 1
                if k in ppl_k:
                    pos = ppl_k.index(k)
                    y = plot_ppl[pos]
                    ax_ppl.annotate(
                        f"-B{blk}",
                        xy=(k, y),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9,
                        fontweight="semibold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="#ff7f0e", linewidth=0.8),
                    )

        ax_ppl.set_yscale("log")
        ax_ppl.set_xlabel("Number of blocks removed (k)", fontsize=11)
        ax_ppl.set_ylabel("Perplexity (log scale)", fontsize=11)
        ax_ppl.set_title("WikiText-2 Perplexity", fontsize=12, pad=10)
        ax_ppl.set_xticks(ppl_k)
        ax_ppl.grid(True, linestyle="--", alpha=0.4)
        ax_ppl.legend(loc="best", framealpha=0.9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02 if has_ppl else 1.0)
    fig.tight_layout()
    return _save(fig, out_path)


def _save(fig, out_path: str) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure to {path}")
    return path

