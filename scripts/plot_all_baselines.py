#!/usr/bin/env python3
"""Plot all baseline and pruning experiment results across all 3 models.

Segregates figures into:
  reports/group9/figures/GPT-2/
  reports/group9/figures/GPT-2-Medium/
  reports/group9/figures/Qwen/
"""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from redundancy.plotting import plot_lm_eval_curve


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_model_files(model_dir: Path):
    """Classify the 5 baseline/pruning files in a model directory."""
    files = sorted(model_dir.glob("*.json"))
    unpruned_file = None
    lowest_bi_file = None
    random_files = {}

    for f in files:
        data = load_json(f)
        pruning = data.get("pruning")
        if not pruning:
            unpruned_file = f
        else:
            # Check filename and pruning run_type
            fname = f.stem.lower()
            if "lowest_bi" in fname:
                lowest_bi_file = f
            elif "random_run0" in fname:
                random_files["random_run0"] = f
            elif "random_run1" in fname:
                random_files["random_run1"] = f
            elif "random_run2" in fname:
                random_files["random_run2"] = f
            else:
                # fallback by run_type
                rt = pruning.get("run_type", "")
                if "lowest_bi" in rt:
                    lowest_bi_file = f
                else:
                    random_files[f.stem] = f

    return unpruned_file, lowest_bi_file, random_files


def extract_data(pruning_file: Path, baseline_file: Path | None = None):
    """Extract k, acc_norm, acc_stderr, ppl, blocks_removed with k=0 baseline aligned."""
    data = load_json(pruning_file)
    pruning = data.get("pruning", {})
    blocks_removed = pruning.get("blocks_removed", [])
    raw_acc = pruning.get("lm_eval", {}).get("acc_norm", [])
    raw_err = pruning.get("lm_eval", {}).get("acc_stderr", [])
    raw_ppl = pruning.get("perplexity", [])

    base_acc, base_err, base_ppl = None, None, None
    if baseline_file and baseline_file.exists():
        base_data = load_json(baseline_file)
        base_ppl = base_data.get("perplexity")
        lm = base_data.get("lm_eval", {})
        if "lm_eval" in lm and isinstance(lm["lm_eval"], dict):
            base_acc = lm["lm_eval"].get("acc_norm,none")
            base_err = lm["lm_eval"].get("acc_stderr,none")

    # In some lowest_bi files (e.g. Qwen, GPT-2-Medium), index 0 is already the unpruned baseline (k=0)
    if len(raw_acc) == len(blocks_removed) + 1:
        # First entry is baseline
        if base_acc is None:
            base_acc = raw_acc[0]
            base_err = raw_err[0]
            base_ppl = raw_ppl[0]
        acc_norm = raw_acc[1:]
        acc_stderr = raw_err[1:]
        ppl = raw_ppl[1:]
    else:
        acc_norm = list(raw_acc)
        acc_stderr = list(raw_err)
        ppl = list(raw_ppl)

    return {
        "model_name": data.get("model_name", "Model"),
        "blocks_removed": blocks_removed,
        "acc_norm": acc_norm,
        "acc_stderr": acc_stderr,
        "perplexity": ppl,
        "base_acc": base_acc,
        "base_err": base_err,
        "base_ppl": base_ppl,
    }


def plot_model_comparison(model_name: str, lowest_data: dict, random_data_dict: dict, out_path: Path):
    """Generate Lowest-BI vs. Random Sweep comparison plot (Accuracy & Perplexity)."""
    fig, (ax_acc, ax_ppl) = plt.subplots(1, 2, figsize=(14, 5.5))

    base_acc = lowest_data.get("base_acc")
    base_err = lowest_data.get("base_err")
    base_ppl = lowest_data.get("base_ppl")

    # 1. Unpruned baseline lines
    if base_acc is not None:
        ax_acc.axhline(base_acc, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8,
                       label=f"Unpruned Baseline ({base_acc:.4f})")
    ax_acc.axhline(0.25, color="#d62728", linestyle=":", linewidth=1.2, alpha=0.7,
                   label="Random chance (0.25)")

    if base_ppl is not None:
        ax_ppl.axhline(base_ppl, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8,
                       label=f"Unpruned Baseline ({base_ppl:.2f})")

    # Determine max_k for comparison (up to max random k, typically 5)
    max_k_compare = 5

    # 2. Plot individual Random runs (faded) and collect matrix
    rand_acc_matrix = []
    rand_ppl_matrix = []
    rand_colors = ["#7f7f7f", "#bcbd22", "#17becf"]

    for i, (r_name, r_data) in enumerate(sorted(random_data_dict.items())):
        k_vals = list(range(1, len(r_data["acc_norm"]) + 1))
        plot_k = [0] + k_vals if base_acc is not None else k_vals
        plot_acc = [base_acc] + r_data["acc_norm"] if base_acc is not None else r_data["acc_norm"]
        plot_ppl = [base_ppl] + r_data["perplexity"] if base_ppl is not None else r_data["perplexity"]

        c = rand_colors[i % len(rand_colors)]
        label_suffix = r_name.replace("baseline_", "").replace("random_run", "Run ")
        ax_acc.plot(plot_k, plot_acc, linestyle="--", marker="o", color=c, alpha=0.5,
                    markersize=4, label=f"Random {label_suffix}")
        ax_ppl.plot(plot_k, plot_ppl, linestyle="--", marker="o", color=c, alpha=0.5,
                    markersize=4, label=f"Random {label_suffix}")

        rand_acc_matrix.append(r_data["acc_norm"][:max_k_compare])
        rand_ppl_matrix.append(r_data["perplexity"][:max_k_compare])

    # 3. Plot Random Mean ± Std
    if rand_acc_matrix:
        rand_acc_arr = np.array(rand_acc_matrix)  # shape (n_runs, max_k)
        rand_ppl_arr = np.array(rand_ppl_matrix)
        mean_acc = np.mean(rand_acc_arr, axis=0)
        std_acc = np.std(rand_acc_arr, axis=0)
        mean_ppl = np.mean(rand_ppl_arr, axis=0)
        std_ppl = np.std(rand_ppl_arr, axis=0)

        k_comp = np.arange(1, len(mean_acc) + 1)
        plot_k_comp = np.concatenate(([0], k_comp)) if base_acc is not None else k_comp
        plot_mean_acc = np.concatenate(([base_acc], mean_acc)) if base_acc is not None else mean_acc
        plot_std_acc = np.concatenate(([0], std_acc)) if base_acc is not None else std_acc

        ax_acc.plot(plot_k_comp, plot_mean_acc, color="#d62728", linewidth=2.5, marker="s",
                    markersize=6, label="Random Mean")
        ax_acc.fill_between(plot_k_comp, plot_mean_acc - plot_std_acc, plot_mean_acc + plot_std_acc,
                            color="#d62728", alpha=0.15, label="Random ± 1 SD")

        plot_mean_ppl = np.concatenate(([base_ppl], mean_ppl)) if base_ppl is not None else mean_ppl
        plot_std_ppl = np.concatenate(([0], std_ppl)) if base_ppl is not None else std_ppl
        ax_ppl.plot(plot_k_comp, plot_mean_ppl, color="#d62728", linewidth=2.5, marker="s",
                    markersize=6, label="Random Mean")
        ax_ppl.fill_between(plot_k_comp, np.maximum(base_ppl * 0.8 if base_ppl is not None else 1.0, plot_mean_ppl - plot_std_ppl),
                            plot_mean_ppl + plot_std_ppl, color="#d62728", alpha=0.15, label="Random ± 1 SD")

    # 4. Plot Lowest-BI (up to max_k_compare for direct comparison, or full curve)
    l_k = list(range(1, len(lowest_data["acc_norm"]) + 1))
    plot_l_k = [0] + l_k if base_acc is not None else l_k
    plot_l_acc = [base_acc] + lowest_data["acc_norm"] if base_acc is not None else lowest_data["acc_norm"]
    plot_l_ppl = [base_ppl] + lowest_data["perplexity"] if base_ppl is not None else lowest_data["perplexity"]

    # Show lowest_bi on the comparable range
    ax_acc.plot(plot_l_k[:max_k_compare + 1], plot_l_acc[:max_k_compare + 1], color="#1f77b4",
                linewidth=2.5, marker="D", markersize=7, label="Lowest-BI Pruning")
    ax_ppl.plot(plot_l_k[:max_k_compare + 1], plot_l_ppl[:max_k_compare + 1], color="#1f77b4",
                linewidth=2.5, marker="D", markersize=7, label="Lowest-BI Pruning")

    # Annotate blocks for lowest_bi
    for idx, blk in enumerate(lowest_data["blocks_removed"][:max_k_compare]):
        k = idx + 1
        acc_val = lowest_data["acc_norm"][idx]
        ax_acc.annotate(f"-B{blk}", (k, acc_val), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, fontweight="bold", color="#1f77b4")

    # Axes styling
    ticks = list(range(0, max_k_compare + 1))
    ax_acc.set_xticks(ticks)
    ax_acc.set_xlabel("Number of blocks removed (k)", fontsize=11)
    ax_acc.set_ylabel("HellaSwag Normalized Accuracy", fontsize=11)
    ax_acc.set_title("HellaSwag Normalized Accuracy (Lowest-BI vs. Random)", fontsize=12, pad=10)
    ax_acc.grid(True, linestyle="--", alpha=0.4)
    ax_acc.legend(loc="best", fontsize=9, framealpha=0.9)

    ax_ppl.set_xticks(ticks)
    ax_ppl.set_yscale("log")
    if base_ppl is not None:
        ax_ppl.set_ylim(bottom=base_ppl * 0.75)
    ax_ppl.set_xlabel("Number of blocks removed (k)", fontsize=11)
    ax_ppl.set_ylabel("WikiText-2 Perplexity (log scale)", fontsize=11)
    ax_ppl.set_title("WikiText-2 Perplexity (Lowest-BI vs. Random)", fontsize=12, pad=10)
    ax_ppl.grid(True, linestyle="--", alpha=0.4)
    ax_ppl.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.suptitle(f"{model_name}: Lowest-BI vs. Random Block Removal Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison to {out_path}")


def main():
    experiments_root = Path("experiments")
    figures_root = Path("reports/group9/figures")

    models = [
        ("GPT-2", "gpt2", "GPT-2 (124M)"),
        ("GPT-2-Medium", "gpt2-medium", "GPT-2-Medium (355M)"),
        ("Qwen", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B (600M)"),
    ]

    for model_folder, model_id, display_name in models:
        model_exp_dir = experiments_root / model_folder
        model_fig_dir = figures_root / model_folder
        model_fig_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n==================================================")
        print(f"Processing Model: {model_folder} -> {model_fig_dir}")
        print(f"==================================================")

        unpruned_file, lowest_bi_file, random_files = find_model_files(model_exp_dir)
        print(f"  Unpruned baseline: {unpruned_file.name if unpruned_file else 'None'}")
        print(f"  Lowest-BI:         {lowest_bi_file.name if lowest_bi_file else 'None'}")
        print(f"  Random runs:       {[f.name for f in random_files.values()]}")

        # 1. Process and plot Lowest-BI
        lowest_data = None
        if lowest_bi_file:
            lowest_data = extract_data(lowest_bi_file, unpruned_file)
            # Focused lm_eval
            plot_lm_eval_curve(
                acc_norm=lowest_data["acc_norm"],
                acc_stderr=lowest_data["acc_stderr"],
                blocks_removed=lowest_data["blocks_removed"],
                out_path=str(model_fig_dir / "lm_eval_lowest_bi.png"),
                title=f"{display_name} — Lowest-BI Pruning HellaSwag Accuracy",
                baseline_acc=lowest_data["base_acc"],
                baseline_stderr=lowest_data["base_err"],
            )
            # Dual panel lm_eval + ppl
            plot_lm_eval_curve(
                acc_norm=lowest_data["acc_norm"],
                acc_stderr=lowest_data["acc_stderr"],
                blocks_removed=lowest_data["blocks_removed"],
                out_path=str(model_fig_dir / "lm_eval_and_ppl_lowest_bi.png"),
                title=f"{display_name} — Lowest-BI Pruning Evaluation",
                baseline_acc=lowest_data["base_acc"],
                baseline_stderr=lowest_data["base_err"],
                perplexity=lowest_data["perplexity"],
                baseline_ppl=lowest_data["base_ppl"],
            )

        # 2. Process and plot each Random run
        random_data_dict = {}
        for r_name, r_file in sorted(random_files.items()):
            r_data = extract_data(r_file, unpruned_file)
            random_data_dict[r_name] = r_data

            clean_name = r_name.replace("baseline_", "")
            # Focused lm_eval
            plot_lm_eval_curve(
                acc_norm=r_data["acc_norm"],
                acc_stderr=r_data["acc_stderr"],
                blocks_removed=r_data["blocks_removed"],
                out_path=str(model_fig_dir / f"lm_eval_{clean_name}.png"),
                title=f"{display_name} — {clean_name} HellaSwag Accuracy",
                baseline_acc=r_data["base_acc"],
                baseline_stderr=r_data["base_err"],
            )
            # Dual panel
            plot_lm_eval_curve(
                acc_norm=r_data["acc_norm"],
                acc_stderr=r_data["acc_stderr"],
                blocks_removed=r_data["blocks_removed"],
                out_path=str(model_fig_dir / f"lm_eval_and_ppl_{clean_name}.png"),
                title=f"{display_name} — {clean_name} Pruning Evaluation",
                baseline_acc=r_data["base_acc"],
                baseline_stderr=r_data["base_err"],
                perplexity=r_data["perplexity"],
                baseline_ppl=r_data["base_ppl"],
            )

        # 3. Model comparison: Lowest-BI vs. Random Runs
        if lowest_data and random_data_dict:
            comp_path = model_fig_dir / "comparison_lowest_bi_vs_random.png"
            plot_model_comparison(display_name, lowest_data, random_data_dict, comp_path)

    # 4. Clean up the loose temporary files generated in reports/group9/figures/ if present
    loose_1 = Path("reports/group9/figures/lm_eval_baseline_Qwen_Qwen3-0.6B_random_run2_20260913_111529.png")
    loose_2 = Path("reports/group9/figures/lm_eval_and_ppl_baseline_Qwen_Qwen3-0.6B_random_run2_20260913_111529.png")
    if loose_1.exists():
        loose_1.unlink()
    if loose_2.exists():
        loose_2.unlink()

    print("\nAll model plots successfully generated and segregated!")


if __name__ == "__main__":
    main()
