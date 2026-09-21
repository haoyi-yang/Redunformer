#!/usr/bin/env python3

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redundancy.data import load_wikitext, prepare_encodings
from redundancy.eval import compute_perplexity
from redundancy.models import clone_and_prune_model, get_device, get_transformer_blocks, load_model
from utils import build_config, build_result_dict, save_results


def parse_args():
    p = argparse.ArgumentParser(description="Remove blocks until a perplexity cap is reached.")
    p.add_argument("--config", type=str, default=None, help="JSON config file.")
    p.add_argument("--measurement", type=str, default=None, help="Path to a measurement JSON with BI scores.")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ppl-cap", type=float, default=None, help="Stop after perplexity exceeds this value. Defaults to 1.2x the unpruned baseline.")
    p.add_argument("--max-removals", type=int, default=None, help="Maximum number of block removals to attempt. Defaults to all removable blocks unless a cap is reached.")
    p.add_argument("--min-distance", type=int, default=3, help="Minimum distance between newly removed blocks for distance-constrained BI selection.")
    p.add_argument("-a", "--all", action="store_true", help="Run all pruning policies.")
    p.add_argument("--policy", nargs="+", default=None, help="One or more policies to compare.")
    p.add_argument("--repeats", type=int, default=3, help="Repeat random policy runs with different seeds.")
    p.add_argument("--skip-lm-eval", action="store_true", help="Compat flag for the shared config builder.")
    p.add_argument("--output-dir", type=str, default="experiments")
    return p.parse_args()


def load_bi_scores(path: str | None) -> list[float]:
    """Load BI scores from a measurement file."""
    if path is None or not Path(path).exists():
        experiments_dir = Path("experiments")
        measurement_files = sorted(experiments_dir.glob("measurement_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not measurement_files:
            raise FileNotFoundError(f"No measurement files found in {experiments_dir}")
        path = str(measurement_files[0])
        print(f"Measurement file not found or not specified. Using newest measurement file: {path}")

    with open(path) as f:
        payload = json.load(f)

    bi_scores = payload.get("bi_scores")
    if not isinstance(bi_scores, list) or not bi_scores:
        raise ValueError(f"No bi_scores found in measurement file: {path}")
    return [float(x) for x in bi_scores]


def load_similarity_matrix(path: str | None):
    """Load the cosine similarity matrix used by zone-based pruning heuristics."""
    if path is None or not Path(path).exists():
        return None

    with open(path) as f:
        payload = json.load(f)

    for key in ("cosine_similarity_matrix", "similarity_matrix", "similarity"):
        matrix = payload.get(key)
        if isinstance(matrix, list):
            return np.asarray(matrix, dtype=float)
    return None


def select_weighted_bi_with_distance(bi_scores, removed=None, n_blocks: int | None = None, max_removals: int | None = None, min_distance: int = 1):
    """Return the full ordered list using weighted BI = BI / (1 + nearest_removed_distance)."""
    if n_blocks is None:
        n_blocks = len(bi_scores)
    removed = [int(x) for x in (removed or [])]
    available = [idx for idx in range(1, n_blocks - 1) if idx not in removed]
    if max_removals is None:
        max_removals = len(available)


    selected = []
    while len(selected) < min(max_removals, len(available)):
        candidates = [idx for idx in available if idx not in selected and all(abs(idx - other) >= max(1, int(min_distance)) for other in removed + selected)]
        if not candidates:
            candidates = [idx for idx in available if idx not in selected]
        if not candidates:
            break

        def score(idx):
            d = min((abs(idx - other) for other in removed + selected), default=10**9)
            return float(bi_scores[idx]) / (1.0 + float(d)), idx

        idx = min(candidates, key=score)
        selected.append(idx)
    return selected


def select_larger_similarity_zone(similarity_matrix, bi_scores, n_blocks: int | None = None, max_removals: int | None = None):
    """Split valid blocks into two clusters using the similarity matrix and keep the larger cluster."""
    if similarity_matrix is None:
        if n_blocks is None:
            n_blocks = len(bi_scores)
        return sorted(range(1, n_blocks - 1), key=lambda idx: (float(bi_scores[idx]), idx))[:max_removals]

    if n_blocks is None:
        n_blocks = similarity_matrix.shape[0]

    valid = list(range(1, n_blocks - 1))
    if not valid:
        return []

    values = []
    for idx in valid:
        row = np.asarray(similarity_matrix[idx], dtype=float)
        others = [row[j] for j in valid if j != idx]
        values.append(float(np.mean(others))) if others else values.append(0.0)

    sorted_pairs = sorted(enumerate(values), key=lambda x: x[1])
    best_gap = -1.0
    best_split = 0
    for i in range(len(sorted_pairs) - 1):
        gap = sorted_pairs[i + 1][1] - sorted_pairs[i][1]
        if gap > best_gap:
            best_gap = gap
            best_split = i + 1

    left = {valid[sorted_pairs[i][0]] for i in range(best_split)}
    right = {valid[sorted_pairs[i][0]] for i in range(best_split, len(sorted_pairs))}
    zone = left if len(left) >= len(right) else right
    ordered = sorted(zone, key=lambda idx: (float(bi_scores[idx]), idx))
    if max_removals is None:
        return ordered
    return ordered[:max_removals]


def select_lowest_bi_with_min_distance(bi_scores, n_blocks: int | None = None, max_removals: int | None = None, min_distance: int = 2):
    """Return the full ordered list of lowest-BI blocks under a minimum spacing constraint."""
    if n_blocks is None:
        n_blocks = len(bi_scores)

    available = range(1, n_blocks - 1)  # Exclude first and last blocks
    removed = []
    min_distance = max(1, int(min_distance))

    if max_removals is None:
        max_removals = len(available)
    # select blocks which has the distance to all other blocks greater than min dist
    while len(removed) < min(max_removals, len(available)):
        candidates = [
            idx for idx in available
            if idx not in removed
            and all(abs(idx - other) >= min_distance for other in removed)
        ]
        if not candidates:
            break

        idx = min(candidates, key=lambda idx: (float(bi_scores[idx]), idx))
        removed.append(idx)
    # select block which has distance to the last entry greater than min dist
    if len(removed) < max_removals:
        last_removed = removed[-1] if removed else None
        candidates = [
            idx for idx in available
            if idx not in removed
            and (last_removed is None or abs(idx - last_removed) >= min_distance)
        ]
        if candidates:
            idx = min(candidates, key=lambda idx: (float(bi_scores[idx]), idx))
            removed.append(idx)
    # final phase, just add the rest in
    while len(removed) < max_removals:
        candidates = [idx for idx in available if idx not in removed]
        if not candidates:
            break
        idx = min(candidates, key=lambda idx: (float(bi_scores[idx]), idx))
        removed.append(idx)
    return removed


def select_greedy_perplexity(model, input_ids, device, stride, candidates, removed=None, compute_fn=None):
    """Choose the next block whose removal minimises perplexity.

    This is intentionally separate from the non-greedy policy selectors, which
    return the full ordered removal list instead of a single-step choice.
    """
    if not candidates:
        raise ValueError("No candidate blocks available for greedy selection.")
    if input_ids is None:
        raise ValueError("input_ids is required for greedy perplexity evaluation.")

    if compute_fn is None:
        compute_fn = compute_perplexity

    kept_removed = [int(x) for x in (removed or [])]
    best_idx = None
    best_ppl = float("inf")

    for idx in candidates:
        trial_removed = kept_removed + [idx]
        trial_model = clone_and_prune_model(model, trial_removed)
        ppl = compute_fn(trial_model, input_ids, device=device, stride=stride)
        if ppl < best_ppl:
            best_ppl = ppl
            best_idx = idx

    if best_idx is None:
        raise ValueError("Greedy selection failed to find a valid candidate.")
    return int(best_idx)


def run_non_greedy_policy_experiment(policy: str, model, input_ids, device, cfg, bi_scores, similarity_matrix=None, max_removals=None, min_distance=1):
    """Run the random/lowest-BI/distance-constrained policies until the cap is hit."""
    start_time = time.perf_counter()
    n_blocks = len(get_transformer_blocks(model))
    if max_removals is None:
        max_removals = max(1, n_blocks - 2)
    cap = 50.0

    if policy == "random":
        random.seed(time.time())
        removed_order = random.sample(range(1, n_blocks - 1), n_blocks - 2)
    elif policy == "lowest_bi":
        valid = list(range(1, n_blocks - 1))
        removed_order = sorted(valid, key=lambda idx: (bi_scores[idx], idx))[: min(max_removals, len(valid))]
    elif policy == "lowest_bi_with_min_distance":
        removed_order = select_lowest_bi_with_min_distance(
            bi_scores,
            n_blocks=n_blocks,
            max_removals=max_removals,
            min_distance=min_distance,
        )
    elif policy == "weighted_bi_distance":
        removed_order = select_weighted_bi_with_distance(
            bi_scores,
            n_blocks=n_blocks,
            max_removals=max_removals,
            min_distance=min_distance,
        )
    elif policy == "larger_similarity_zone":
        removed_order = select_larger_similarity_zone(
            similarity_matrix,
            bi_scores,
            n_blocks=n_blocks,
            max_removals=max_removals,
        )
    else:
        raise ValueError(f"Unknown non-greedy policy: {policy}")

    removed = []
    perplexities = []
    stop_reason = "max_removals"

    for step, idx in enumerate(removed_order, start=1):
        removed.append(int(idx))
        pruned_model = clone_and_prune_model(model, removed)
        ppl = compute_perplexity(pruned_model, input_ids, device=device, stride=cfg["stride"])
        perplexities.append(float(ppl))

        print(f"[policy={policy}] step={step} removed_block={idx} ppl={ppl:.2f} cap={cap:.2f}")
        if ppl > cap:
            stop_reason = "ppl_cap"
            break

    elapsed_seconds = time.perf_counter() - start_time
    print(f"[policy={policy}] elapsed_seconds={elapsed_seconds:.3f}")

    result = (
        policy,
        {
            "blocks_removed": removed[: len(perplexities)],
            "perplexity_at_step": perplexities,
            "total_removed": int(len(perplexities)),
            "elapsed_seconds": float(elapsed_seconds),
        },
    )
    return result


def run_greedy_perplexity_experiment(model, input_ids, device, cfg, max_removals=None):
    """Greedy policy: choose the next block that minimizes perplexity, stop at cap."""
    start_time = time.perf_counter()
    n_blocks = len(get_transformer_blocks(model))
    if max_removals is None:
        max_removals = max(1, n_blocks - 2)

    cap = 50

    removed = []
    removed_order = []
    perplexities = []
    stop_reason = "max_removals"
    candidates = list(range(1, n_blocks - 1))

    while len(removed_order) < max_removals and candidates:
        available = [idx for idx in candidates if idx not in removed]
        if not available:
            break

        idx = select_greedy_perplexity(model, input_ids, device, cfg["stride"], available, removed)
        removed.append(int(idx))
        removed_order.append(int(idx))
        candidates.remove(idx)

        pruned_model = clone_and_prune_model(model, removed)
        ppl = compute_perplexity(pruned_model, input_ids, device=device, stride=cfg["stride"])
        perplexities.append(float(ppl))

        print(f"[policy=greedy_perplexity] step={len(removed_order)} removed_block={idx} ppl={ppl:.2f} cap={cap:.2f}")
        if ppl > cap:
            stop_reason = "ppl_cap"
            break

    elapsed_seconds = time.perf_counter() - start_time
    print(f"[policy=greedy_perplexity] elapsed_seconds={elapsed_seconds:.3f}")

    result = (
        "greedy_perplexity",
        {
            "blocks_removed": removed_order,
            "perplexity_at_step": perplexities,
            "total_removed": int(len(removed_order)),
            "elapsed_seconds": float(elapsed_seconds),
        },
    )
    return result


def main():
    args = parse_args()
    cfg = build_config(args)

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = get_device(cfg["device"])
    model, tokenizer = load_model(cfg["model_name"], device=str(device))

    dataset = load_wikitext(split="test")
    input_ids = prepare_encodings(dataset, tokenizer, cfg["max_length"], cfg["stride"])

    measurement_path = args.measurement if args.measurement else None
    bi_scores = load_bi_scores(measurement_path)
    similarity_matrix = load_similarity_matrix(measurement_path)
    if len(bi_scores) != len(get_transformer_blocks(model)):
        raise ValueError(
            f"Measurement/model mismatch: measurement has {len(bi_scores)} BI scores, but {cfg['model_name']} has {len(get_transformer_blocks(model))} blocks."
        )

    all_policies = ["random", "lowest_bi", "lowest_bi_with_min_distance", "weighted_bi_distance", "greedy_perplexity"]
    requested_policies = all_policies if args.all else args.policy

    final_results = []

    # Run and save each policy result as its own report (one file per policy/run)
    if "random" in requested_policies:
        for i in range(args.repeats):
            rng = random.Random(cfg["seed"] + i)
            label = f"random_run{i}"
            print(f"\n=== Running policy: {label} ===")
            result = run_non_greedy_policy_experiment("random", model, input_ids, device, cfg, bi_scores, similarity_matrix=similarity_matrix, max_removals=args.max_removals, min_distance=args.min_distance)
            policy_name, data = result
            out = build_result_dict(cfg, device)
            out.update({"policy": label, "result": data})
            save_results(out, cfg["output_dir"], f"{cfg['model_name']}_{label}")
            final_results.append((label, data))

    for policy in requested_policies:
        if policy == "random":
            continue
        if policy == "greedy_perplexity":
            label = "greedy_perplexity"
            print(f"\n=== Running policy: {label} ===")
            result = run_greedy_perplexity_experiment(model, input_ids, device, cfg, max_removals=args.max_removals)
            policy_name, data = result
            out = build_result_dict(cfg, device)
            out.update({"policy": label, "result": data})
            save_results(out, cfg["output_dir"], f"{cfg['model_name']}_{label}")
            final_results.append((label, data))
            continue

        print(f"\n=== Running policy: {policy} ===")
        result = run_non_greedy_policy_experiment(policy, model, input_ids, device, cfg, bi_scores, similarity_matrix=similarity_matrix, max_removals=args.max_removals, min_distance=args.min_distance)
        policy_name, data = result
        out = build_result_dict(cfg, device)
        out.update({"policy": policy, "result": data})
        save_results(out, cfg["output_dir"], f"{cfg['model_name']}_{policy}")
        final_results.append((policy, data))

    return final_results


if __name__ == "__main__":
    main()
