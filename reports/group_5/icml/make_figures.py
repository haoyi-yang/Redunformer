"""Figures for the Group 5 ICML-style report.

Depth, task, replacement, and LoRA numbers are the values printed in paper.tex.
The Qwen3.5-9B curve is read from Henrik's pruning JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NINE_B = (
    ROOT
    / "experiments"
    / "results"
    / "recovery_meta-llama__Llama-2-7b-hf_20260915T053928Z"
    / "pruning_Qwen__Qwen3.5-9B_20260915T004903Z.json"
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
GRAY = "#7F7F7F"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _grouped(ax, labels, series, ylabel, *, ylim=None, hline=None):
    x = np.arange(len(labels))
    n = len(series)
    width = 0.8 / n
    for i, (name, values, color) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        vals = np.array(values, dtype=float)
        ax.bar(x + offset, vals, width * 0.92, label=name, color=color, zorder=2)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if hline is not None:
        ax.axhline(hline, color="#333333", lw=0.6, zorder=1)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(frameon=False)


def plot_9b(path: Path) -> None:
    payload = json.loads(NINE_B.read_text(encoding="utf-8"))
    baseline = float(payload["baseline"]["perplexity"])
    by_strategy: dict[str, list[tuple[float, float]]] = {
        "importance": [(0.0, 1.0)],
        "frequency": [(0.0, 1.0)],
        "random": [(0.0, 1.0)],
    }
    for run in payload["runs"]:
        by_strategy[run["strategy"]].append(
            (float(run["ratio"]) * 100.0, float(run["perplexity"]) / baseline)
        )
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    styles = {
        "importance": (ORANGE, "-", "o"),
        "frequency": (BLUE, "-.", "s"),
        "random": (GRAY, "--", None),
    }
    for name in ("importance", "frequency", "random"):
        pts = sorted(by_strategy[name])
        color, ls, marker = styles[name]
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=color,
            ls=ls,
            marker=marker,
            ms=3.5,
            lw=1.6,
            label=name,
        )
    ax.axhline(1.0, color="#333333", lw=0.8, ls=":", label="unmasked")
    ax.set_yscale("log")
    ax.set_xlabel("neurons masked per layer (%)")
    ax.set_ylabel("perplexity / unmasked")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_depth(path: Path) -> None:
    # Raw WikiText-2 perplexity change at 10% masking, same numbers as Table 2.
    raw = {
        "GPT-2\n124M": (5.24, 4.08, 2.31),
        "GPT-2\n355M": (7.06, 2.92, 1.77),
        "GPT-2\n774M": (1.06, 0.98, 1.01),
        "Qwen3\n0.6B": (2.69, 0.19, 0.81),
        "Qwen3.5\n4B": (0.34, 0.37, 1.02),
        "Qwen3.5\n9B": (0.16, 0.14, 0.57),
        "Llama-2\n7B": (0.27, 0.10, 0.42),
    }
    labels = list(raw)
    early, middle, deep = [], [], []
    for triple in raw.values():
        scale = max(triple)
        early.append(triple[0] / scale)
        middle.append(triple[1] / scale)
        deep.append(triple[2] / scale)
    fig, ax = plt.subplots(figsize=(6.9, 2.7))
    _grouped(
        ax,
        labels,
        [
            ("early", early, BLUE),
            ("middle", middle, ORANGE),
            ("deep", deep, GREEN),
        ],
        "cost / worst band in that model",
        ylim=(0, 1.18),
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_task(path: Path) -> None:
    # PIQA accuracy at 25% masking. Same cells as the task table.
    labels = ["Llama-2\n7B", "Qwen3.5\n4B", "Qwen3\n0.6B", "GPT-2\n124M"]
    wiki = [0.750, 0.705, 0.594, 0.562]
    piqa = [0.785, 0.755, 0.630, 0.544]
    random = [0.695, 0.675, 0.594, 0.570]
    fig, ax = plt.subplots(figsize=(3.35, 2.7))
    _grouped(
        ax,
        labels,
        [
            ("ranked on WikiText", wiki, BLUE),
            ("ranked on PIQA", piqa, ORANGE),
            ("random", random, GRAY),
        ],
        "PIQA accuracy",
        ylim=(0.50, 0.85),
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_replacement(path: Path) -> None:
    labels = ["GPT-2\n124M", "Qwen3\n0.6B", "Llama-2\n7B"]
    masked = [1.16, 2.24, 0.04]
    after = [-0.02, 0.28, 0.02]
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    _grouped(
        ax,
        labels,
        [
            ("mask one twin", masked, VERMILLION),
            ("route through partner", after, GREEN),
        ],
        r"$\Delta$ WikiText-2 perplexity",
        hline=0,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_recovery(path: Path) -> None:
    labels = [
        "GPT-2\n124M",
        "GPT-2\n355M",
        "GPT-2\n774M",
        "Qwen3\n0.6B",
        "Llama-2\n7B",
        "Qwen3.5\n4B*",
    ]
    ten = [80, 84, 76, 76, -6, 2]
    twenty_five = [87, 89, 80, 83, 48, np.nan]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    _grouped(
        ax,
        labels,
        [
            ("10% masked", ten, BLUE),
            ("25% masked", twenty_five, ORANGE),
        ],
        "masking gap recovered (%)",
        hline=0,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    plot_9b(HERE / "qwen35_9b_masking.png")
    plot_depth(HERE / "depth_bands.png")
    plot_task(HERE / "task_piqa.png")
    plot_replacement(HERE / "replacement.png")
    plot_recovery(HERE / "lora_recovery.png")


if __name__ == "__main__":
    main()
