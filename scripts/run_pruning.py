import argparse
import json
import os
import random
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from redundancy.models import load_model_and_tokenizer
from redundancy.data import load_evaluation_dataset, load_calibration_dataset
from redundancy.eval import (
    evaluate_perplexity,
    build_divergence_probe,
    dense_reference_outputs,
    output_divergence,
)
from redundancy.hooks import collect_gram_stats
from redundancy.scoring import wanda_tile_scores, sparsegpt_tile_errors
from redundancy.recovery import reconstruct_prune_tiles
from redundancy.combined import prune_wanda_recon, prune_random_recon
from redundancy.policy import load_sensitivity, classify, expand_to_model, allocate


# Methods whose result depends on a random seed. random_recon reproduces `random`'s tile
# choice at the same seed, so it must be seeded and filed exactly like random -- gating on
# the literal string "random" left it with seed=None (RNG falling back to system entropy,
# so the tiles did NOT match) and no _seed suffix, so all five seeds overwrote one file.
SEEDED_METHODS = ("random", "random_recon")

MATRICES = {
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
}


def get_full_tiles(weight, tile_size):
    rows, cols = weight.shape
    tiles = []

    for r in range(0, rows, tile_size):
        for c in range(0, cols, tile_size):
            tile = weight[r:r + tile_size, c:c + tile_size]
            if tile.shape == (tile_size, tile_size):
                tiles.append((r, c))

    return tiles


def zero_tiles(weight, tiles_to_prune, tile_size):
    with torch.no_grad():
        for r, c in tiles_to_prune:
            weight[r:r + tile_size, c:c + tile_size] = 0


def extract_pruned_mask(weight, original_weight, tile_size, rel_tol=0.1):
    """Return [[row, col], ...] tile coords that pruning removed (vectorised, non-invasive).

    A tile counts as removed if its max-abs magnitude collapsed below rel_tol x its
    original. This is exact for the maskers (the tile is hard zero) and robust for
    reconstruction, where bf16/inversion rounding leaves a tiny residual instead of a
    hard zero.
    """
    with torch.no_grad():
        rows, cols = weight.shape
        nr, nc = rows // tile_size, cols // tile_size
        cut_r, cut_c = nr * tile_size, nc * tile_size
        wt = weight[:cut_r, :cut_c].contiguous().float().reshape(nr, tile_size, nc, tile_size)
        w0 = original_weight[:cut_r, :cut_c].contiguous().float().reshape(nr, tile_size, nc, tile_size)
        cur = wt.abs().amax(dim=3).amax(dim=1)     # [nr, nc] current max-abs per tile
        orig = w0.abs().amax(dim=3).amax(dim=1)     # [nr, nc] original max-abs per tile
        removed = (orig > 0) & (cur <= rel_tol * orig)
        idx = removed.nonzero(as_tuple=False)
    return [[int(i) * tile_size, int(j) * tile_size] for i, j in idx.tolist()]


def prune_lowest_magnitude(weight, tile_size, prune_ratio):
    tiles = get_full_tiles(weight, tile_size)
    scored_tiles = []

    for r, c in tiles:
        tile = weight[r:r + tile_size, c:c + tile_size]
        score = torch.norm(tile).item()
        scored_tiles.append((score, r, c))

    scored_tiles.sort(key=lambda x: x[0])

    num_prune = int(len(scored_tiles) * prune_ratio)
    tiles_to_prune = [(r, c) for _, r, c in scored_tiles[:num_prune]]

    zero_tiles(weight, tiles_to_prune, tile_size)

    return len(tiles), num_prune


def prune_highest_magnitude(weight, tile_size, prune_ratio):
    tiles = get_full_tiles(weight, tile_size)
    scored_tiles = []

    for r, c in tiles:
        tile = weight[r:r + tile_size, c:c + tile_size]
        score = torch.norm(tile).item()
        scored_tiles.append((score, r, c))

    scored_tiles.sort(key=lambda x: x[0], reverse=True)

    num_prune = int(len(scored_tiles) * prune_ratio)
    tiles_to_prune = [(r, c) for _, r, c in scored_tiles[:num_prune]]

    zero_tiles(weight, tiles_to_prune, tile_size)

    return len(tiles), num_prune


def prune_random(weight, tile_size, prune_ratio, seed):
    tiles = get_full_tiles(weight, tile_size)

    rng = random.Random(seed)
    rng.shuffle(tiles)

    num_prune = int(len(tiles) * prune_ratio)
    tiles_to_prune = tiles[:num_prune]

    zero_tiles(weight, tiles_to_prune, tile_size)

    return len(tiles), num_prune


def prune_by_scores(weight, scored_tiles, tile_size, prune_ratio):
    """Prune the lowest-scoring tiles. scored_tiles is a list of (score, r, c)."""
    scored_tiles = sorted(scored_tiles, key=lambda x: x[0])

    num_prune = int(len(scored_tiles) * prune_ratio)
    tiles_to_prune = [(r, c) for _, r, c in scored_tiles[:num_prune]]

    zero_tiles(weight, tiles_to_prune, tile_size)

    return len(scored_tiles), num_prune


def prune_wanda(weight, tile_size, prune_ratio, col_norms):
    scored = wanda_tile_scores(weight, col_norms, tile_size)
    return prune_by_scores(weight, scored, tile_size, prune_ratio)


def prune_sparsegpt(weight, tile_size, prune_ratio, hessian):
    scored = sparsegpt_tile_errors(weight, hessian, tile_size)
    return prune_by_scores(weight, scored, tile_size, prune_ratio)


def prune_sparsegpt_recon(weight, tile_size, prune_ratio, hessian):
    # Ranks by eq. 23 error AND applies the compensating weight updates in place.
    return reconstruct_prune_tiles(weight, hessian, tile_size, prune_ratio)


def apply_pruning(weight, method, tile_size, prune_ratio, seed, stat=None):
    if method == "magnitude":
        return prune_lowest_magnitude(weight, tile_size, prune_ratio)
    if method == "magnitude_high":
        return prune_highest_magnitude(weight, tile_size, prune_ratio)
    if method == "random":
        return prune_random(weight, tile_size, prune_ratio, seed)
    if method == "wanda":
        return prune_wanda(weight, tile_size, prune_ratio, stat)
    if method == "sparsegpt":
        return prune_sparsegpt(weight, tile_size, prune_ratio, stat)
    if method == "sparsegpt_recon":
        return prune_sparsegpt_recon(weight, tile_size, prune_ratio, stat)
    if method == "wanda_recon":
        # Wanda's selection + SparseGPT's repair: isolates selection from reconstruction.
        # stat is the Gram matrix H; Wanda's column norms are sqrt(diag(H)).
        return prune_wanda_recon(weight, tile_size, prune_ratio, stat)
    if method == "random_recon":
        # Random selection + SparseGPT's repair. Picks the SAME tiles as `random` at the same
        # seed, so random vs random_recon isolates repair, and random_recon vs wanda_recon
        # isolates selection. stat is the Gram matrix H.
        return prune_random_recon(weight, tile_size, prune_ratio, stat, seed)

    raise ValueError(f"Unknown pruning method: {method}")


def get_target_weight(model, target_name):
    for name, param in model.named_parameters():
        if name == target_name:
            return param

    raise ValueError(f"Could not find target weight: {target_name}")


def get_target_module(model, target_name):
    """Resolve 'model.layers.0.mlp.up_proj.weight' -> the up_proj nn.Linear module."""
    module_path = target_name[:-len(".weight")] if target_name.endswith(".weight") else target_name

    module = model
    for attr in module_path.split("."):
        module = module[int(attr)] if attr.isdigit() else getattr(module, attr)

    return module


def collect_stats_for_targets(model, calib_samples, target_names, method):
    """Collect the calibration statistic each metric needs, for every target
    matrix, in a single calibration pass. Returns {target_name: stat} where the
    stat is the Wanda column-norm vector or the SparseGPT Gram matrix H."""
    modules_by_name = {name: get_target_module(model, name) for name in target_names}
    collectors = collect_gram_stats(model, modules_by_name, calib_samples)

    stats = {}
    for name, collector in collectors.items():
        if method == "wanda":
            stats[name] = collector.col_norms().detach().clone()
        else:  # sparsegpt keeps the full Gram matrix
            stats[name] = collector.H

    return stats


def get_num_layers(model):
    layer_ids = set()

    for name, _ in model.named_parameters():
        parts = name.split(".")
        if len(parts) > 3 and parts[0] == "model" and parts[1] == "layers":
            if parts[2].isdigit():
                layer_ids.add(int(parts[2]))

    if not layer_ids:
        raise ValueError("Could not automatically detect model layers.")

    return max(layer_ids) + 1


def build_target_name(layer, matrix_name):
    component = MATRICES[matrix_name]
    return f"model.layers.{layer}.{component}.{matrix_name}.weight"


def ratio_short_name(prune_ratio):
    return str(int(prune_ratio * 100))


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Results saved to {path}")


def run_single_experiment(model, tokenizer, dataset, args, target_name, seed=None, stat=None,
                          probe_ids=None, reference=None):
    target_weight = get_target_weight(model, target_name)
    original_weight = target_weight.detach().clone()

    print("\n=======================================")
    print(f"Pruning target matrix: {target_name}")
    print(f"Matrix shape: {target_weight.shape}")
    print(f"Tile size: {args.tile_size}")
    print(f"Prune ratio: {args.prune_ratio}")
    print(f"Method: {args.method}")
    print(f"Seed: {seed if args.method == 'random' else None}")
    print("=======================================")

    num_tiles, num_pruned = apply_pruning(
        target_weight,
        args.method,
        args.tile_size,
        args.prune_ratio,
        seed,
        stat=stat,
    )

    print(f"Total full tiles: {num_tiles}")
    print(f"Pruned tiles: {num_pruned}")

    ppl = evaluate_perplexity(model, tokenizer, dataset, eval_frac=args.eval_frac)

    print(f"Final Perplexity after pruning: {ppl:.4f}")

    divergence = None
    if probe_ids is not None and reference is not None:
        divergence = output_divergence(model, probe_ids, reference)
        print(
            f"Divergence vs dense: KL={divergence['kl_dense_pruned']:.4f} "
            f"top1={divergence['top1_agreement']:.3f} cos={divergence['hidden_cosine']:.4f}"
        )

    # Record the pruning mask (best-effort; must never break the experiment).
    try:
        pruned_mask = extract_pruned_mask(target_weight, original_weight, args.tile_size)
    except Exception as exc:
        print(f"[warn] mask extraction failed: {exc}")
        pruned_mask = None

    with torch.no_grad():
        target_weight.copy_(original_weight)

    return {
        "target_matrix": target_name,
        "tile_size": args.tile_size,
        "prune_ratio": args.prune_ratio,
        "num_tiles": num_tiles,
        "num_pruned": num_pruned,
        "method": args.method,
        "seed": seed if args.method in SEEDED_METHODS else None,
        "perplexity": ppl,
        "divergence": divergence,
        "pruned_mask": pruned_mask,
    }


def run_layer_experiment(model, tokenizer, dataset, args, layer, seed=None,
                         layer_stats=None, probe_ids=None, reference=None):
    """Prune ALL seven matrices of one layer simultaneously, evaluate once, restore.

    Measures the cumulative effect of pruning a whole layer (Rathore Strategy 2),
    as opposed to the per-matrix sensitivity of run_single_experiment.
    """
    targets = {m: build_target_name(layer, m) for m in MATRICES.keys()}
    weights = {m: get_target_weight(model, t) for m, t in targets.items()}
    originals = {m: w.detach().clone() for m, w in weights.items()}

    print("\n=======================================")
    print(f"Whole-layer pruning: layer {layer}  method {args.method}  ratio {args.prune_ratio}")
    print("=======================================")

    matrix_info = []
    for matrix_name in MATRICES.keys():
        stat = layer_stats[targets[matrix_name]] if layer_stats is not None else None
        num_tiles, num_pruned = apply_pruning(
            weights[matrix_name], args.method, args.tile_size, args.prune_ratio, seed, stat=stat,
        )
        try:
            mask = extract_pruned_mask(weights[matrix_name], originals[matrix_name], args.tile_size)
        except Exception as exc:
            print(f"[warn] mask extraction failed ({matrix_name}): {exc}")
            mask = None
        matrix_info.append({
            "matrix": matrix_name,
            "num_tiles": num_tiles,
            "num_pruned": num_pruned,
            "pruned_mask": mask,
        })

    ppl = evaluate_perplexity(model, tokenizer, dataset, eval_frac=args.eval_frac)
    print(f"Whole-layer perplexity: {ppl:.4f}")

    divergence = None
    if probe_ids is not None and reference is not None:
        divergence = output_divergence(model, probe_ids, reference)
        print(
            f"Divergence vs dense: KL={divergence['kl_dense_pruned']:.4f} "
            f"top1={divergence['top1_agreement']:.3f} cos={divergence['hidden_cosine']:.4f}"
        )

    with torch.no_grad():
        for matrix_name in MATRICES.keys():
            weights[matrix_name].copy_(originals[matrix_name])

    return {
        "layer": layer,
        "method": args.method,
        "seed": seed if args.method in SEEDED_METHODS else None,
        "tile_size": args.tile_size,
        "prune_ratio": args.prune_ratio,
        "scope": "whole_layer",
        "perplexity": ppl,
        "divergence": divergence,
        "matrices": matrix_info,
    }


def build_tile_counts(model, layers, tile_size):
    """{(layer, matrix): number of full tiles} -- needed to budget-match a policy."""
    tiles = {}
    for layer in layers:
        for matrix_name in MATRICES.keys():
            w = get_target_weight(model, build_target_name(layer, matrix_name))
            rows, cols = w.shape
            tiles[(layer, matrix_name)] = (rows // tile_size) * (cols // tile_size)
    return tiles


def run_wholemodel_experiment(model, tokenizer, dataset, args, layers, seed=None,
                              calib_samples=None, probe_ids=None, reference=None,
                              ratio_map=None):
    """Prune every matrix across ALL layers at one uniform sparsity, evaluate once.

    Layers are processed in order; for the data-aware methods each layer's calibration
    is collected on the model AS IT CURRENTLY STANDS (earlier layers already pruned),
    i.e. the faithful sequential one-shot approach. Each layer's stats are freed before
    the next, so peak memory is one layer's Gram matrices. There is no restore -- the
    returned model is the fully pruned model (run one sparsity per process).
    """
    needs_calib = args.method in ("wanda", "sparsegpt", "sparsegpt_recon", "wanda_recon", "random_recon")
    total_tiles = 0
    total_pruned = 0
    per_layer = []

    # --matrices restricts whole-model pruning to specific projection types, pruned SIMULTANEOUSLY
    # across every layer (no restore). Distinct from --all-matrices screening, which prunes one
    # matrix, evaluates, and restores -- that measures each in isolation, not their composition.
    scan_matrices = getattr(args, "matrices", None) or list(MATRICES.keys())

    for layer in layers:
        layer_stats = None
        if needs_calib:
            target_names = [build_target_name(layer, m) for m in scan_matrices]
            layer_stats = collect_stats_for_targets(model, calib_samples, target_names, args.method)

        layer_pruned = 0
        for matrix_name in scan_matrices:
            target_name = build_target_name(layer, matrix_name)
            weight = get_target_weight(model, target_name)
            stat = layer_stats[target_name] if layer_stats is not None else None
            # Policy A (uniform) uses one ratio everywhere; Policy B supplies a per-matrix ratio.
            ratio = ratio_map[(layer, matrix_name)] if ratio_map is not None else args.prune_ratio
            num_tiles, num_pruned = apply_pruning(
                weight, args.method, args.tile_size, ratio, seed, stat=stat,
            )
            total_tiles += num_tiles
            total_pruned += num_pruned
            layer_pruned += num_pruned

        per_layer.append({"layer": layer, "num_pruned": layer_pruned})
        print(f"  layer {layer:2}: pruned {layer_pruned} tiles (total {total_pruned})")

        del layer_stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Evaluating fully pruned whole model...")
    ppl = evaluate_perplexity(model, tokenizer, dataset, eval_frac=args.eval_frac)
    print(f"Whole-model perplexity: {ppl:.4f}")

    divergence = None
    if probe_ids is not None and reference is not None:
        divergence = output_divergence(model, probe_ids, reference)
        print(
            f"Divergence vs dense: KL={divergence['kl_dense_pruned']:.4f} "
            f"top1={divergence['top1_agreement']:.3f} cos={divergence['hidden_cosine']:.4f}"
        )

    return {
        "method": args.method,
        "seed": seed if args.method in SEEDED_METHODS else None,
        "tile_size": args.tile_size,
        "prune_ratio": args.prune_ratio,
        "policy": getattr(args, "policy", "uniform"),
        "scope": "whole_model",
        "num_layers": len(layers),
        "total_tiles": total_tiles,
        "total_pruned": total_pruned,
        "achieved_sparsity": (total_pruned / total_tiles) if total_tiles else None,
        "perplexity": ppl,
        "divergence": divergence,
        "per_layer": per_layer,
    }


def load_layer_json_files(experiment_dir):
    data = []

    for filename in os.listdir(experiment_dir):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(experiment_dir, filename)

        with open(path, "r") as f:
            item = json.load(f)

        if "results" in item and "layer" in item and "method" in item:
            data.append(item)

    return data


def make_plots_from_json(experiment_dir, baseline_ppl):
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    all_data = load_layer_json_files(experiment_dir)

    if not all_data:
        raise ValueError(f"No valid JSON files found in {experiment_dir}")

    grouped = defaultdict(list)

    for item in all_data:
        method = item["method"]
        seed = item.get("seed", None)

        if method == "random":
            key = f"random_seed{seed}"
        else:
            key = method

        grouped[key].append(item)

    for key, summaries in grouped.items():
        summaries = sorted(summaries, key=lambda x: x["layer"])
        layers = [s["layer"] for s in summaries]

        # Only plot the matrices actually present. A scan restricted with --matrices (as the
        # cluster analysis does) has a subset, and assuming all seven raises KeyError *after*
        # every result is already on disk -- turning a cosmetic plotting failure into a
        # non-zero exit that looks identical to a real one.
        present = [m for m in MATRICES.keys() if all(
            any(r["matrix"] == m for r in s["results"]) for s in summaries)]
        matrix_to_values = {m: [] for m in present}

        for summary in summaries:
            result_map = {r["matrix"]: r["perplexity"] for r in summary["results"]}
            for matrix in present:
                matrix_to_values[matrix].append(result_map[matrix])

        if not matrix_to_values:
            continue

        plt.figure(figsize=(15, 7), dpi=200)

        for matrix, values in matrix_to_values.items():
            plt.plot(layers, values, marker="o", linewidth=1.8, label=matrix)

        plt.axhline(
            baseline_ppl,
            linestyle="--",
            linewidth=1.2,
            label=f"Baseline PPL = {baseline_ppl}",
        )

        plt.title(f"{key}: perplexity across all layers")
        plt.xlabel("Layer")
        plt.ylabel("Perplexity")
        plt.xticks(layers)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()

        out = os.path.join(plot_dir, f"{key}_all_layers_projection_trend.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()

        print(f"Saved plot: {out}")

    mean_by_group = {}
    n_matrices = 0

    for key, summaries in grouped.items():
        values = []

        for summary in summaries:
            layer = summary["layer"]
            n_matrices = max(n_matrices, len(summary["results"]))
            mean_ppl = sum(r["perplexity"] for r in summary["results"]) / len(summary["results"])
            values.append((layer, mean_ppl))

        mean_by_group[key] = sorted(values, key=lambda x: x[0])

    plt.figure(figsize=(15, 7), dpi=200)

    for key, values in mean_by_group.items():
        layers = [x[0] for x in values]
        mean_ppl = [x[1] for x in values]
        plt.plot(layers, mean_ppl, marker="o", linewidth=2, label=key)

    plt.axhline(
        baseline_ppl,
        linestyle="--",
        linewidth=1.2,
        label=f"Baseline PPL = {baseline_ppl}",
    )

    # Title and filename must name the methods actually plotted. These were hardcoded to
    # "magnitude vs random", so every run -- wanda, sparsegpt, anything -- emitted a plot
    # captioned "magnitude vs random" over data from a completely different method. The y-label
    # likewise claimed 7 matrices even when --matrices restricts the scan to fewer.
    methods_plotted = " vs ".join(sorted(mean_by_group))
    plt.title(f"Mean layer sensitivity: {methods_plotted}")
    plt.xlabel("Layer")
    plt.ylabel(f"Mean perplexity across {n_matrices} projection matri"
               f"{'x' if n_matrices == 1 else 'ces'}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(plot_dir, "mean_layer_trend.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out}")

    magnitude_key = None
    random_keys = []

    for key in grouped.keys():
        if key == "magnitude":
            magnitude_key = key
        if key.startswith("random_seed"):
            random_keys.append(key)

    if magnitude_key is not None and random_keys:
        for random_key in random_keys:
            mag_map = {}
            rand_map = {}

            for summary in grouped[magnitude_key]:
                layer = summary["layer"]
                for result in summary["results"]:
                    mag_map[(layer, result["matrix"])] = result["perplexity"]

            for summary in grouped[random_key]:
                layer = summary["layer"]
                for result in summary["results"]:
                    rand_map[(layer, result["matrix"])] = result["perplexity"]

            common_layers = sorted(set(l for l, _ in mag_map.keys()) & set(l for l, _ in rand_map.keys()))

            for matrix in MATRICES.keys():
                layers = []
                mag_vals = []
                rand_vals = []

                for layer in common_layers:
                    key_pair = (layer, matrix)
                    if key_pair in mag_map and key_pair in rand_map:
                        layers.append(layer)
                        mag_vals.append(mag_map[key_pair])
                        rand_vals.append(rand_map[key_pair])

                if not layers:
                    continue

                plt.figure(figsize=(12, 6), dpi=200)
                plt.plot(layers, mag_vals, marker="o", linewidth=2, label="Magnitude")
                plt.plot(layers, rand_vals, marker="o", linewidth=2, label=random_key)

                plt.axhline(
                    baseline_ppl,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Baseline PPL = {baseline_ppl}",
                )

                plt.title(f"{matrix}: magnitude vs {random_key}")
                plt.xlabel("Layer")
                plt.ylabel("Perplexity")
                plt.xticks(layers)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()

                out = os.path.join(
                    plot_dir,
                    f"{matrix}_magnitude_vs_{random_key}_layer_trend.png",
                )
                plt.savefig(out, bbox_inches="tight")
                plt.close()
                print(f"Saved plot: {out}")

            diff_layers = common_layers
            diff_matrix = []

            for matrix in MATRICES.keys():
                row = []
                for layer in diff_layers:
                    key_pair = (layer, matrix)
                    if key_pair in mag_map and key_pair in rand_map:
                        row.append(mag_map[key_pair] - rand_map[key_pair])
                    else:
                        row.append(float("nan"))
                diff_matrix.append(row)

            diff_matrix = np.array(diff_matrix)

            plt.figure(figsize=(16, 6), dpi=200)
            im = plt.imshow(diff_matrix, aspect="auto")

            plt.colorbar(im, label="Magnitude PPL - Random PPL")
            plt.yticks(range(len(MATRICES)), list(MATRICES.keys()))
            plt.xticks(range(len(diff_layers)), diff_layers)
            plt.xlabel("Layer")
            plt.ylabel("Projection matrix")
            plt.title(f"Difference heatmap: magnitude - {random_key}")

            for i in range(diff_matrix.shape[0]):
                for j in range(diff_matrix.shape[1]):
                    val = diff_matrix[i, j]
                    if not np.isnan(val):
                        plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

            plt.tight_layout()

            out = os.path.join(
                plot_dir,
                f"magnitude_minus_{random_key}_difference_heatmap.png",
            )
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {out}")


def main():
    parser = argparse.ArgumentParser(description="Run tile-level pruning experiment.")

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")

    parser.add_argument("--tile-size", type=int, default=32)  # project standard (Rathore §2.4); 64 was a bug
    parser.add_argument("--prune-ratio", type=float, default=0.20)

    parser.add_argument(
        "--method",
        type=str,
        choices=["magnitude", "magnitude_high", "random", "wanda", "sparsegpt", "sparsegpt_recon",
                 "wanda_recon", "random_recon"],
        default="magnitude",
    )

    parser.add_argument(
        "--calib-samples",
        type=int,
        default=128,
        help="Number of calibration windows for wanda/sparsegpt.",
    )

    parser.add_argument(
        "--calib-seqlen",
        type=int,
        default=512,
        help="Token length of each calibration window.",
    )

    parser.add_argument(
        "--eval-frac",
        type=float,
        default=1.0,
        help="Fraction of the eval corpus for perplexity (e.g. 0.2 for fast screening; 1.0 = full).",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds, e.g. --seeds 123 456 789",
    )

    parser.add_argument(
        "--target-name",
        type=str,
        default="model.layers.0.mlp.up_proj.weight",
        help="Single target matrix to prune when --all-matrices is not used.",
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to run, e.g. --layers 0 12 27",
    )

    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Run every transformer layer detected in the model.",
    )

    parser.add_argument(
        "--all-matrices",
        action="store_true",
        help="Run all seven projection matrices for each layer.",
    )

    parser.add_argument(
        "--matrices",
        type=str,
        nargs="+",
        default=None,
        choices=list(MATRICES.keys()),
        help="With --all-matrices, restrict the scan to these projection types "
             "(e.g. --matrices o_proj up_proj). Default: all seven. Used by the cluster "
             "analysis, which scans only the most robust and most sensitive matrix types.",
    )

    parser.add_argument(
        "--whole-layer",
        action="store_true",
        help="Prune all seven matrices of each layer simultaneously and evaluate once.",
    )

    parser.add_argument(
        "--whole-model",
        action="store_true",
        help="Prune every matrix across all layers at one uniform sparsity, evaluate once.",
    )

    parser.add_argument(
        "--policy",
        type=str,
        choices=["uniform", "sensitivity"],
        default="uniform",
        help="Whole-model budget policy: uniform (A), or sensitivity-aware budget-matched (B).",
    )

    parser.add_argument(
        "--screen-dir",
        type=str,
        default="experiments/screen",
        help="Screening results used to derive the sensitivity classes for --policy sensitivity.",
    )

    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments/full_scan",
        help="Directory where JSON files and plots will be saved.",
    )

    parser.add_argument(
        "--baseline-ppl",
        type=float,
        default=13.2181,
        help="Baseline perplexity shown as a reference line in plots.",
    )

    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only create plots from existing JSON files.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for single-matrix mode.",
    )

    args = parser.parse_args()

    if args.plot_only:
        make_plots_from_json(args.experiment_dir, args.baseline_ppl)
        return

    model, tokenizer = load_model_and_tokenizer(args.model)
    dataset = load_evaluation_dataset(args.dataset, args.subset, split="test")

    # Dense reference for output-divergence, computed once on the fully dense model.
    print("Building divergence probe and dense reference...")
    probe_ids = build_divergence_probe(tokenizer, dataset)
    reference = dense_reference_outputs(model, probe_ids)

    needs_calibration = args.method in ("wanda", "sparsegpt", "sparsegpt_recon", "wanda_recon", "random_recon")
    calib_samples = None
    if needs_calibration:
        calib_samples = load_calibration_dataset(
            tokenizer,
            n_samples=args.calib_samples,
            seqlen=args.calib_seqlen,
        )

    if args.all_layers:
        num_layers = get_num_layers(model)
        layers = list(range(num_layers))
        print(f"Detected {num_layers} layers: {layers}")
    elif args.layers is not None:
        layers = args.layers
    else:
        layers = None

    if args.whole_layer:
        if layers is None or len(layers) == 0:
            raise ValueError("Use --layers or --all-layers with --whole-layer")

        seeds_to_run = args.seeds if args.method in SEEDED_METHODS else [None]

        for seed in seeds_to_run:
            for layer in layers:
                layer_stats = None
                if needs_calibration:
                    target_names = [build_target_name(layer, m) for m in MATRICES.keys()]
                    print(f"Collecting {args.method} calibration stats for layer {layer}...")
                    layer_stats = collect_stats_for_targets(model, calib_samples, target_names, args.method)

                result = run_layer_experiment(
                    model, tokenizer, dataset, args, layer, seed=seed,
                    layer_stats=layer_stats, probe_ids=probe_ids, reference=reference,
                )

                ratio = ratio_short_name(args.prune_ratio)
                if args.method in SEEDED_METHODS:
                    filename = f"layer{layer}_wholelayer_{args.method}_p{ratio}_seed{seed}.json"
                else:
                    filename = f"layer{layer}_wholelayer_{args.method}_p{ratio}.json"

                summary = {
                    "model": args.model,
                    "dataset": args.dataset,
                    "subset": args.subset,
                    "calib_samples": args.calib_samples if needs_calibration else None,
                    "calib_seqlen": args.calib_seqlen if needs_calibration else None,
                    "eval_frac": args.eval_frac,
                    **result,
                }
                save_json(os.path.join(args.experiment_dir, filename), summary)

        return

    if args.whole_model:
        if layers is None:
            layers = list(range(get_num_layers(model)))
        print(f"Whole-model pruning over {len(layers)} layers: {layers}")

        # Policy B: derive per-matrix budgets from the screening map, budget-matched
        # so the total tiles removed matches uniform at the same target sparsity.
        ratio_map, policy_info = None, None
        if args.policy == "sensitivity":
            sens = load_sensitivity(args.screen_dir, method="sparsegpt_recon", ref_ratio="0.20")
            classes = classify(sens)
            measured = sorted({l for (l, _) in sens})
            full = expand_to_model(classes, layers, list(MATRICES.keys()), measured)
            tiles = build_tile_counts(model, layers, args.tile_size)
            ratio_map, policy_info = allocate(full, tiles, args.prune_ratio)
            print(f"Policy B (sensitivity-aware, budget-matched) — measured layers {measured}")
            print(f"  target sparsity {policy_info['target']:.3f}  ->  achieved {policy_info['achieved']:.4f}")
            for c, d in policy_info["per_class"].items():
                print(f"  {c:10} ratio {d['ratio']:.3f}   matrices {d['matrices']:3}   tiles {d['tiles']:,}")

        seed = args.seeds[0] if args.method in SEEDED_METHODS else None
        result = run_wholemodel_experiment(
            model, tokenizer, dataset, args, layers, seed=seed,
            calib_samples=calib_samples, probe_ids=probe_ids, reference=reference,
            ratio_map=ratio_map,
        )

        ratio = ratio_short_name(args.prune_ratio)
        suffix = f"_seed{seed}" if args.method in SEEDED_METHODS else ""
        pol = "" if args.policy == "uniform" else f"_{args.policy}"
        summary = {
            "model": args.model,
            "dataset": args.dataset,
            "subset": args.subset,
            "calib_samples": args.calib_samples if needs_calibration else None,
            "calib_seqlen": args.calib_seqlen if needs_calibration else None,
            "eval_frac": args.eval_frac,
            "policy_info": policy_info,
            **result,
        }
        save_json(os.path.join(args.experiment_dir, f"wholemodel_{args.method}_p{ratio}{pol}{suffix}.json"), summary)
        return

    if args.all_matrices:
        if layers is None or len(layers) == 0:
            raise ValueError("Use --layers or --all-layers with --all-matrices")

        seeds_to_run = args.seeds if args.method in SEEDED_METHODS else [None]

        for seed in seeds_to_run:
            for layer in layers:
                layer_results = []

                print("\n#######################################")
                print(f"Running Layer {layer}")
                print(f"Method: {args.method}")
                print(f"Seed: {seed if args.method == 'random' else None}")
                print("#######################################")

                # --matrices restricts the scan to specific projection types. The cluster analysis
                # (Rathore W4/S4) is built around this: "Do not automatically repeat all seven
                # matrix tests in every neighbouring layer. Instead, choose the most sensitive
                # projection matrix ... the most robust projection matrix ... Run these two matrix
                # types across both clusters."
                scan_matrices = args.matrices or list(MATRICES.keys())

                layer_stats = None
                if needs_calibration:
                    target_names = [build_target_name(layer, m) for m in scan_matrices]
                    print(f"Collecting {args.method} calibration stats for layer {layer}...")
                    layer_stats = collect_stats_for_targets(
                        model, calib_samples, target_names, args.method
                    )

                for matrix_name in scan_matrices:
                    target_name = build_target_name(layer, matrix_name)
                    stat = layer_stats[target_name] if layer_stats is not None else None

                    result = run_single_experiment(
                        model=model,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        args=args,
                        target_name=target_name,
                        seed=seed,
                        stat=stat,
                        probe_ids=probe_ids,
                        reference=reference,
                    )

                    result["layer"] = layer
                    result["matrix"] = matrix_name
                    layer_results.append(result)

                ratio = ratio_short_name(args.prune_ratio)

                if args.method in SEEDED_METHODS:
                    filename = f"layer{layer}_{args.method}_p{ratio}_seed{seed}.json"
                else:
                    filename = f"layer{layer}_{args.method}_p{ratio}.json"

                output_path = os.path.join(args.experiment_dir, filename)

                layer_summary = {
                    "model": args.model,
                    "dataset": args.dataset,
                    "subset": args.subset,
                    "layer": layer,
                    "method": args.method,
                    "seed": seed if args.method in SEEDED_METHODS else None,
                    "tile_size": args.tile_size,
                    "prune_ratio": args.prune_ratio,
                    "calib_samples": args.calib_samples if needs_calibration else None,
                    "calib_seqlen": args.calib_seqlen if needs_calibration else None,
                    "eval_frac": args.eval_frac,
                    "results": layer_results,
                }

                save_json(output_path, layer_summary)

        make_plots_from_json(args.experiment_dir, args.baseline_ppl)

    else:
        seed = args.seeds[0] if args.method in SEEDED_METHODS else None

        stat = None
        if needs_calibration:
            print(f"Collecting {args.method} calibration stats for {args.target_name}...")
            stats = collect_stats_for_targets(
                model, calib_samples, [args.target_name], args.method
            )
            stat = stats[args.target_name]

        result = run_single_experiment(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            args=args,
            target_name=args.target_name,
            seed=seed,
            stat=stat,
            probe_ids=probe_ids,
            reference=reference,
        )

        output_path = args.output
        if output_path is None:
            output_path = "experiments/tile_pruning_results.json"

        results = {
            "model": args.model,
            "dataset": args.dataset,
            "subset": args.subset,
            **result,
        }

        save_json(output_path, results)


if __name__ == "__main__":
    main()