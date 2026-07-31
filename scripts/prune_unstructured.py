"""
Unstructured (1x1) pruning -- the tile-size = 1 ceiling for the tile-size study.

At T=1 every scorer in the tile loop degenerates to standard per-weight
unstructured pruning, but the tile loop cannot physically reach it (100.9M tiles
per layer => 3.6B .item() GPU syncs model-wide). This script is the vectorized
1x1 workstream (plan Stage 2D): build a boolean weight mask by top-k on |W| or
|W|*col_norms, optionally repair the survivors with standard unstructured
SparseGPT (per-output-row OBS, NOT the tile-grouping reconstruction), then
evaluate on the same downstream tasks (HellaSwag / PIQA / ARC-Easy) as the tile
ladder so the point drops straight onto the C(T) curve.

Methods
  magnitude              mask = smallest |W|                 (mask-only unless --repair)
  wanda                  mask = smallest |W|*col_norms       (mask-only unless --repair)
  unstructured_sparsegpt SparseGPT saliency mask + OBS repair (mask+repair jointly)

Repair (--repair sparsegpt) is the standard SparseGPT column-sweep: left-to-right
OBS over the columns, all output rows in parallel, O(in^2 * out) and NO per-row
inverse -- which is what makes 1x1 tractable on 3.6B weights. Each pruned weight is
compensated using the surviving columns to its RIGHT, so the sweep equals the exact
per-row least-squares optimum when the mask leaves full right-compensation and is a
bounded sequential approximation otherwise -- exactly the standard/literature
SparseGPT behaviour for an externally supplied mask (this sequential approximation
is precisely why SparseGPT is used instead of a per-row joint inverse). The helper
exact_row_reference() gives the joint per-row LS optimum for validation only; it is
intractable at Qwen scale for unstructured masks (a distinct dense inverse per row),
which is why it is NOT the shipped repair.

The comparison group for the mask is per output row by default (Wanda's canonical
group; yields the target sparsity exactly in every matrix); --comparison global
ranks over the whole matrix instead.

CPU self-test (--selftest) validates the masking math (exact sparsity, wanda
column protection) and that the SparseGPT repair matches the exact per-row
least-squares reference. It does NOT touch the GPU.

Example
  python scripts/prune_unstructured.py --method wanda --prune-ratio 0.50 \
    --repair sparsegpt --tasks hellaswag piqa arc_easy \
    --output experiments/tilesize/downstream/us_wanda_recon_p50_T1.json
"""

import argparse
import gc
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                    # scripts/  (reuse run_pruning helpers)
sys.path.insert(0, os.path.join(HERE, "..", "src"))        # src/

import numpy as np


# ----------------------------------------------------------------------------
# Masking (numpy/torch-agnostic core works on numpy; torch wrappers below)
# ----------------------------------------------------------------------------

def _row_topk_mask(metric, ratio):
    """Boolean PRUNE mask (True = zeroed): smallest `ratio` of each row by `metric`.
    Exact per-row count => achieved sparsity == round(ratio*cols)/cols per row."""
    rows, cols = metric.shape
    k = int(round(ratio * cols))
    if k <= 0:
        return np.zeros_like(metric, dtype=bool)
    if k >= cols:
        return np.ones_like(metric, dtype=bool)
    # kth-smallest threshold per row via partition; prune the k smallest.
    kth = np.partition(metric, k - 1, axis=1)[:, k - 1:k]           # (rows,1)
    mask = metric <= kth
    # partition can leave >k under <= ties; trim extras deterministically per row.
    over = mask.sum(axis=1) - k
    if np.any(over > 0):
        order = np.argsort(metric, axis=1, kind="stable")           # ascending
        keep = np.zeros_like(mask)
        rowidx = np.arange(rows)[:, None]
        keep[rowidx, order[:, :k]] = True
        mask = keep
    return mask


def _global_topk_mask(metric, ratio):
    """Boolean PRUNE mask over the whole matrix: smallest `ratio` entries."""
    n = metric.size
    k = int(round(ratio * n))
    if k <= 0:
        return np.zeros_like(metric, dtype=bool)
    if k >= n:
        return np.ones_like(metric, dtype=bool)
    thresh = np.partition(metric.reshape(-1), k - 1)[k - 1]
    mask = metric <= thresh
    over = int(mask.sum()) - k
    if over > 0:                                                    # trim ties deterministically
        flat_order = np.argsort(metric.reshape(-1), kind="stable")
        keep = np.zeros(n, dtype=bool)
        keep[flat_order[:k]] = True
        mask = keep.reshape(metric.shape)
    return mask


def build_mask(metric, ratio, comparison):
    return _row_topk_mask(metric, ratio) if comparison == "row" else _global_topk_mask(metric, ratio)


# ----------------------------------------------------------------------------
# SparseGPT unstructured repair (torch) -- exact LS reconstruction for a mask
# ----------------------------------------------------------------------------

def sparsegpt_unstructured(weight, hessian, sparsity, prune_mask=None,
                           damp=1e-2, blocksize=128):
    """Standard SparseGPT column-sweep. weight (out,in), hessian (in,in).

    prune_mask given  -> zero exactly those entries, exact LS repair of survivors.
    prune_mask None   -> adaptive per-column-block saliency mask at `sparsity`.

    Returns the pruned+repaired weight (float32, same shape). Not in place.
    """
    import torch
    W = weight.detach().float().clone()
    H = hessian.detach().float().clone().to(W.device)
    rows, cols = W.shape

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    damp_val = damp * torch.mean(torch.diag(H))
    diag = torch.arange(cols, device=W.device)
    H[diag, diag] += damp_val

    # Inverse via Cholesky, then the upper-triangular Cholesky factor of the
    # inverse (the standard SparseGPT parameterization: Hinv[j,j] is the OBS denom).
    L = torch.linalg.cholesky(H)
    Hinv_full = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv_full, upper=True)

    mask_t = None
    if prune_mask is not None:
        mask_t = torch.as_tensor(prune_mask, dtype=torch.bool, device=W.device)

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        if mask_t is not None:
            mask1 = mask_t[:, i1:i2]
        else:                                                       # adaptive saliency mask
            dinv = torch.diag(Hinv1).reshape(1, -1)
            tmp = (W1 ** 2) / (dinv ** 2)
            k = int(round(count * sparsity))
            mask1 = torch.zeros_like(W1, dtype=torch.bool)
            if k > 0:
                thresh = torch.sort(tmp, dim=1).values[:, k - 1:k]
                mask1 = tmp <= thresh

        for j in range(count):
            w = W1[:, j]
            d = Hinv1[j, j]
            q = w.clone()
            q[mask1[:, j]] = 0.0
            Q1[:, j] = q
            err = (w - q) / d
            W1[:, j:] -= err.unsqueeze(1) * Hinv1[j, j:].unsqueeze(0)
            Err1[:, j] = err

        W[:, i1:i2] = Q1
        if i2 < cols:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return W


def exact_row_reference(weight, hessian, prune_mask, damp=1e-2):
    """Reference: the EXACT joint per-row least-squares optimum for a FIXED mask
    (redundancy.recovery math applied per output row -- compensates using ALL
    surviving columns, not just those to the right). This is the true minimum of
    ||X W^T - X W_hat^T||^2 subject to W_hat[mask]=0, so it lower-bounds the
    sequential SparseGPT sweep's error. Used only to validate the sweep in the
    self-test; a separate dense inverse per row makes it intractable at Qwen scale
    for unstructured masks. weight (out,in), hessian (in,in)."""
    import torch
    W = weight.detach().float().clone()
    H = hessian.detach().float().clone().to(W.device)
    rows, cols = W.shape
    diag_mean = torch.diag(H).mean()
    idx = torch.arange(cols, device=W.device)
    Hd = H.clone()
    Hd[idx, idx] += damp * diag_mean
    Hinv = torch.linalg.inv(Hd)
    out = W.clone()
    mask_t = torch.as_tensor(prune_mask, dtype=torch.bool, device=W.device)
    for i in range(rows):
        P = torch.nonzero(mask_t[i], as_tuple=False).flatten()
        if P.numel() == 0:
            continue
        S = torch.linalg.inv(Hinv[P][:, P])
        A = Hinv[:, P] @ S                                          # (cols, |P|)
        wP = W[i, P]
        out[i, :] = W[i, :] - A @ wP
    return out


# ----------------------------------------------------------------------------
# col-norms from Gram (Wanda metric)
# ----------------------------------------------------------------------------

def col_norms_np(H):
    import torch
    return torch.sqrt(torch.clamp(torch.diag(H.float()), min=0.0)).cpu().numpy()


# ----------------------------------------------------------------------------
# Whole-model application
# ----------------------------------------------------------------------------

def metric_for(method, W_np, colnorms):
    if method == "magnitude":
        return np.abs(W_np)
    if method == "wanda":
        return np.abs(W_np) * colnorms[None, :]
    raise ValueError(f"no explicit metric for method {method}")


def prune_matrix(weight, H, method, ratio, comparison, repair, damp, blocksize):
    """Apply unstructured pruning to one weight tensor IN PLACE.
    Returns (num_weights, num_pruned)."""
    import torch
    rows, cols = weight.shape
    n = rows * cols

    if method == "unstructured_sparsegpt":
        orig_zeros = int((weight.detach() == 0).sum().item())
        newW = sparsegpt_unstructured(weight, H, ratio, prune_mask=None,
                                      damp=damp, blocksize=blocksize)
        pruned = int((newW == 0).sum().item()) - orig_zeros   # net new hard zeros
        with torch.no_grad():
            weight.copy_(newW.to(weight.dtype))
        return n, max(pruned, 0)

    colnorms = col_norms_np(H) if method == "wanda" else None
    W_np = weight.detach().float().cpu().numpy()
    metric = metric_for(method, W_np, colnorms)
    mask = build_mask(metric, ratio, comparison)                   # True = prune
    num_pruned = int(mask.sum())

    if repair == "sparsegpt":
        newW = sparsegpt_unstructured(weight, H, ratio, prune_mask=mask,
                                      damp=damp, blocksize=blocksize)
        with torch.no_grad():
            weight.copy_(newW.to(weight.dtype))
    else:                                                          # mask-only
        with torch.no_grad():
            m = torch.as_tensor(mask, device=weight.device)
            weight[m] = 0.0
    return n, num_pruned


def main():
    p = argparse.ArgumentParser(description="Unstructured (1x1) pruning ceiling for the tile study.")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--method", default="wanda",
                   choices=["magnitude", "wanda", "unstructured_sparsegpt"])
    p.add_argument("--prune-ratio", type=float, default=0.50)
    p.add_argument("--comparison", choices=["row", "global"], default="row",
                   help="Mask comparison group: per output row (default) or whole matrix.")
    p.add_argument("--repair", choices=["none", "sparsegpt"], default=None,
                   help="Survivor repair. Default: none for magnitude/wanda, sparsegpt is "
                        "implied for unstructured_sparsegpt.")
    p.add_argument("--damp", type=float, default=1e-2)
    p.add_argument("--blocksize", type=int, default=128)
    p.add_argument("--layers", type=int, nargs="+", default=None)
    p.add_argument("--matrices", nargs="+", default=None)
    p.add_argument("--calib-samples", type=int, default=64)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--tasks", nargs="+", default=["hellaswag", "piqa", "arc_easy"])
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--eval-ppl", action="store_true",
                   help="Evaluate WikiText-2 test perplexity (same eval as the tile ladder) "
                        "instead of downstream lm-eval. Cheap; used for the 1x1-vs-tile ppl screen.")
    p.add_argument("--eval-frac", type=float, default=1.0,
                   help="Fraction of the perplexity eval set to use (matches run_pruning --eval-frac).")
    p.add_argument("--output", default=None)
    p.add_argument("--selftest", action="store_true",
                   help="CPU-only: validate masking + SparseGPT-repair math on synthetic data.")
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    import torch
    from run_pruning import (MATRICES, build_target_name, get_target_weight,
                             collect_stats_for_targets, get_num_layers, save_json)
    from redundancy.models import load_model_and_tokenizer
    from redundancy.data import load_calibration_dataset

    repair = args.repair
    if repair is None:
        repair = "sparsegpt" if args.method == "unstructured_sparsegpt" else "none"
    needs_calib = (args.method in ("wanda", "unstructured_sparsegpt")) or (repair == "sparsegpt")

    matrices = args.matrices or list(MATRICES.keys())
    model, tokenizer = load_model_and_tokenizer(args.model)
    layers = args.layers if args.layers is not None else list(range(get_num_layers(model)))

    calib = None
    if needs_calib:
        calib = load_calibration_dataset(tokenizer, n_samples=args.calib_samples,
                                         seqlen=args.calib_seqlen)

    total_w = total_pruned = 0
    for li in layers:
        names = [build_target_name(li, m) for m in matrices]
        stats = None
        if needs_calib:
            # method != "wanda" here returns the full Gram H (wanda repair also needs H).
            stats = collect_stats_for_targets(model, calib, names, "sparsegpt_recon")
        for m, tname in zip(matrices, names):
            w = get_target_weight(model, tname)
            H = stats[tname] if stats is not None else None
            nw, npd = prune_matrix(w, H, args.method, args.prune_ratio, args.comparison,
                                   repair, args.damp, args.blocksize)
            total_w += nw
            total_pruned += npd
        del stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  layer {li:2}: pruned {total_pruned}/{total_w}")

    achieved = total_pruned / total_w if total_w else 0.0
    print(f"Unstructured {args.method} (repair={repair}) @ {args.prune_ratio}: "
          f"achieved {achieved:.4f}")

    del calib
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    out = {
        "model": args.model,
        "method": args.method,
        "repair": repair,
        "comparison": args.comparison,
        "prune_ratio": args.prune_ratio,
        "tile_size": 1,
        "achieved_sparsity": achieved,
        "total_weights": total_w,
        "total_pruned": total_pruned,
        # Provenance: record calibration config so 1x1 files are self-documenting and
        # comparable to the 32x32 ladder (the 128-vs-64 calib confound hid here before).
        "calib_samples": args.calib_samples if needs_calib else None,
        "calib_seqlen": args.calib_seqlen if needs_calib else None,
    }

    if args.eval_ppl:
        # WikiText-2 test perplexity -- the SAME eval the tile ladder uses, so the
        # 1x1 point drops straight onto the C(T) perplexity curve (no lm-eval).
        from redundancy.data import load_evaluation_dataset
        from redundancy.eval import evaluate_perplexity
        print(f"Evaluating WikiText-2 test perplexity (eval_frac={args.eval_frac}) ...")
        dataset = load_evaluation_dataset()
        ppl = evaluate_perplexity(model, tokenizer, dataset, eval_frac=args.eval_frac)
        tag = f"unstructured_{args.method}_repair-{repair}_p{int(args.prune_ratio*100)}_T1_ppl"
        out.update({"tag": tag, "eval_frac": args.eval_frac, "perplexity": ppl})
        path = args.output or os.path.join("experiments", "tilesize", "ppl", f"{tag}.json")
        save_json(path, out)
        print(f"\n=== perplexity (unstructured / T=1 / p={args.prune_ratio}) ===")
        print(f"  WikiText-2 test perplexity: {ppl:.4f}")
    else:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        print(f"Running lm-eval on tasks: {args.tasks}")
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
        res = lm_eval.simple_evaluate(model=lm, tasks=args.tasks, limit=args.limit)
        tag = f"unstructured_{args.method}_repair-{repair}_p{int(args.prune_ratio*100)}_T1"
        out.update({"tag": tag, "limit": args.limit, "results": res["results"]})
        path = args.output or os.path.join("experiments", "tilesize", "downstream", f"{tag}.json")
        save_json(path, out)
        print("\n=== downstream accuracy (unstructured / T=1) ===")
        for t, v in res["results"].items():
            acc = v.get("acc_norm,none", v.get("acc,none"))
            print(f"  {t:12} {acc}")


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

def selftest():
    import torch
    print("=" * 72)
    print("  SELF-TEST: unstructured masking + SparseGPT-repair math (CPU)")
    print("=" * 72)
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        if not cond:
            fails += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    rng = np.random.default_rng(0)

    # ---- masking: exact per-row sparsity ---------------------------------
    W = rng.standard_normal((16, 20)).astype(np.float32)
    for r in (0.05, 0.10, 0.25, 0.50):
        mask = build_mask(np.abs(W), r, "row")
        per_row = mask.sum(axis=1)
        k = int(round(r * 20))
        check(f"row mask exact per-row count at r={r}", np.all(per_row == k),
              f"per_row={set(per_row.tolist())} expected {k}")

    # ---- masking: prunes the actual smallest |W| in a row ----------------
    row = np.array([5, 1, 4, 2, 3, 9, 8, 7, 6, 0], dtype=np.float32)[None, :]
    mask = build_mask(np.abs(row), 0.3, "row")                     # k=3 -> zero {0,1,2}
    zeroed = set(np.nonzero(mask[0])[0].tolist())
    check("row mask picks the 3 smallest magnitudes", zeroed == {9, 1, 3},
          f"zeroed idx={zeroed}")   # values 0,1,2 sit at columns 9,1,3

    # ---- masking: wanda column protection --------------------------------
    Ww = np.ones((4, 4), dtype=np.float32) * 0.5
    Ww[:, 0] = 0.4                                                  # slightly smaller |W| in col 0
    colnorms = np.array([10.0, 1.0, 1.0, 1.0], dtype=np.float32)   # col 0 has huge activation norm
    metric = np.abs(Ww) * colnorms[None, :]
    mask_mag = build_mask(np.abs(Ww), 0.25, "row")                 # magnitude prunes col 0 (smallest |W|)
    mask_wanda = build_mask(metric, 0.25, "row")                   # wanda protects col 0
    check("magnitude prunes the small-|W| column 0", np.all(mask_mag[:, 0]))
    check("wanda protects column 0 (large activation norm)", not np.any(mask_wanda[:, 0]),
          f"wanda pruned cols per row = {[np.nonzero(mask_wanda[i])[0].tolist() for i in range(4)]}")

    # ---- masking: global count exact -------------------------------------
    mg = build_mask(np.abs(W), 0.25, "global")
    check("global mask exact total count", int(mg.sum()) == int(round(0.25 * W.size)),
          f"got {int(mg.sum())} expected {int(round(0.25*W.size))}")

    # ---- SparseGPT repair correctness invariants -------------------------
    # Use CORRELATED features so repair genuinely matters (with iid X, H is ~
    # diagonal and no compensation is possible).
    torch.manual_seed(0)
    rows, cols = 12, 24
    Wt = torch.randn(rows, cols).double()
    Acov = torch.randn(cols, cols).double()
    cov = Acov @ Acov.t() / cols + 0.1 * torch.eye(cols).double()
    X = torch.randn(600, cols).double() @ torch.linalg.cholesky(cov).t()
    H = X.t() @ X
    def recon_err(Wc):
        return ((X @ Wt.t() - X @ Wc.double().t()) ** 2).sum().item()

    mask = build_mask(Wt.abs().numpy(), 0.5, "row")
    m_t = torch.as_tensor(mask)
    sweep = sparsegpt_unstructured(Wt, H, 0.5, prune_mask=mask, damp=1e-2, blocksize=8).double()
    check("SparseGPT repair zeros exactly the masked entries",
          torch.all(sweep[m_t].abs() < 1e-9), f"max|masked|={sweep[m_t].abs().max().item():.2e}")

    maskonly = Wt.clone(); maskonly[m_t] = 0.0
    e_mask, e_rep = recon_err(maskonly), recon_err(sweep)
    check("SparseGPT repair reduces output reconstruction error vs mask-only",
          e_rep < e_mask, f"mask-only={e_mask:.3e}  repaired={e_rep:.3e}")

    # exact joint LS lower-bounds the sequential sweep (sweep can't beat the optimum).
    ref = exact_row_reference(Wt, H, mask, damp=1e-2).double()
    e_ref = recon_err(ref)
    check("exact joint-LS reference <= sequential sweep error <= mask-only error",
          e_ref <= e_rep + 1e-6 and e_rep <= e_mask + 1e-6,
          f"exact_joint={e_ref:.3e}  sweep={e_rep:.3e}  mask-only={e_mask:.3e}")

    # When the mask prunes only the leftmost columns, full right-compensation is
    # available and the sequential sweep MUST equal the exact joint optimum --
    # this numerically validates the OBS column-update math (single- and cross-block).
    lmask = np.zeros((rows, cols), dtype=bool); lmask[:, :12] = True
    for bs in (cols, 8):
        sweepL = sparsegpt_unstructured(Wt, H, 0.5, prune_mask=lmask, damp=1e-2, blocksize=bs).double()
        refL = exact_row_reference(Wt, H, lmask, damp=1e-2).double()
        relL = ((sweepL - refL).norm() / (refL.norm() + 1e-12)).item()
        check(f"sweep == exact joint LS for left-compensated mask (blocksize={bs}, rel<1e-4)",
              relL < 1e-4, f"rel_diff={relL:.2e}")

    print("=" * 72)
    print(f"  SELF-TEST RESULT: {'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
    print("=" * 72)
    return fails == 0


if __name__ == "__main__":
    main()
