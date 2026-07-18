# Codebase Documentation

## Overview

This project evaluates **GPT-2** on **WikiText-2** to establish a baseline for block/layer-level redundancy experiments. The pipeline loads the model, tokenises the dataset with a sliding window, computes perplexity, and optionally runs [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) benchmarks.

---

## Pipeline Flow

```
run_baseline.py
  │
  ├─ utils.build_config()           → merge JSON config + CLI args
  ├─ models.load_model()            → GPT-2 model + tokenizer on best device
  ├─ data.load_wikitext()           → WikiText-2 test split
  ├─ data.prepare_encodings()       → sliding-window token chunks
  ├─ eval.compute_perplexity()      → perplexity score
  ├─ eval.run_lm_eval()             → (optional) HellaSwag accuracy
  ├─ utils.build_result_dict()      → structured results dict
  ├─ utils.save_results()           → experiments/baseline_gpt2_<timestamp>.json
  └─ utils.print_summary()          → final summary to stdout
```

---

## Module Reference

### `src/redundancy/models.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_device` | `(device: str \| None) → torch.device` | Picks the best available device. Priority: user-specified → CUDA → MPS → CPU. |
| `load_model` | `(model_name, device, dtype) → (model, tokenizer)` | Downloads the model and tokenizer from HuggingFace, moves to device, sets eval mode. Defaults to `gpt2` with `float16` on GPU/MPS and `float32` on CPU. |

**Where is GPT-2 stored?** The model weights are **not** kept in this repository. When `load_model("gpt2")` runs for the first time, HuggingFace Transformers downloads the weights (~525 MB) from the Hub and caches them at `~/.cache/huggingface/hub/models--gpt2/`. Every subsequent run loads directly from that cache — no re-download needed. This keeps the git repo lightweight.

---

### `src/redundancy/data.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_wikitext` | `(split) → Dataset` | Loads `Salesforce/wikitext` (`wikitext-2-raw-v1`) from HuggingFace Hub. |
| `prepare_encodings` | `(dataset, tokenizer, max_length, stride) → Tensor` | Joins all text, tokenises it into one long sequence, then slices it into overlapping windows of `max_length` tokens with a step of `stride`. Returns a `(n_windows, max_length)` tensor. |

**Why sliding windows?** GPT-2 has a fixed context of 1024 tokens. To evaluate on a longer text, we slide a window across the full token sequence. Overlapping regions give the model context, but we only score the *non-overlapping* tokens to avoid counting any token twice.

---

### `src/redundancy/eval.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_perplexity` | `(model, input_ids, device, stride) → float` | Iterates over all sliding windows, runs a forward pass on each, computes cross-entropy loss on the non-overlapping portion, and returns `exp(avg_loss)`. Prints progress every 50 windows. |
| `run_lm_eval` | `(model_name, tasks, device, batch_size) → dict` | Calls `lm_eval.simple_evaluate` with the `"hf"` backend. Returns the full results dict. Used for standardised benchmarks like HellaSwag. |

**Perplexity scoring detail:** For window `i > 0`, only the last `stride` tokens are scored (the earlier tokens are overlap from the previous window). For window `i = 0`, all tokens are scored. This follows the [HuggingFace perplexity guide](https://huggingface.co/docs/transformers/perplexity).

---

### `src/utils/__init__.py`

| Function | Description |
|----------|-------------|
| `load_json_config` | Loads a JSON config file and returns a dict. |
| `build_config` | Merges a JSON config file with CLI argument overrides. Hardcodes `model_name="gpt2"` and `dataset="wikitext-2-raw-v1"`. |
| `build_result_dict` | Builds a structured results dict with perplexity, config, device, and system info (platform, Python version, PyTorch version). |
| `save_results` | Writes the results dict to `experiments/baseline_gpt2_<timestamp>.json`. |
| `print_summary` | Prints a final summary block with model name, perplexity, and lm-eval scores. |

---

### `scripts/run_baseline.py`

| Function | Description |
|----------|-------------|
| `parse_args` | Defines CLI arguments: `--config`, `--max-length`, `--stride`, `--seed`, `--device`, `--skip-lm-eval`, `--output-dir`, `--batch-size`. |
| `main` | Orchestrates the full pipeline: parse args → load model → load data → compute perplexity → (optional) lm-eval → save → print summary. |


---

## Configuration

`configs/baseline.json` contains default parameters:

| Key | Default | Description |
|-----|---------|-------------|
| `model_name` | `"gpt2"` | HuggingFace model identifier |
| `dataset` | `"wikitext-2-raw-v1"` | WikiText-2 raw subset |
| `max_length` | `1024` | Context window size (tokens) |
| `stride` | `512` | Sliding window step size |
| `lm_eval_tasks` | `["hellaswag"]` | Benchmark tasks for lm-eval-harness |
| `batch_size` | `4` | Batch size for lm-eval |
| `seed` | `42` | Random seed |

---

## Output Format

Each run produces a JSON file in `experiments/`:

```json
{
  "timestamp": "2026-06-10T09:01:17+00:00",
  "model_name": "gpt2",
  "dataset": "wikitext-2-raw-v1",
  "max_length": 1024,
  "stride": 512,
  "seed": 42,
  "device": "mps",
  "system": {
    "platform": "macOS-...",
    "python": "3.13.5",
    "torch": "2.12.0"
  },
  "perplexity": 25.18,
  "lm_eval": null
}
```

When `--skip-lm-eval` is not set, `lm_eval` contains per-task metrics (e.g. `acc_norm` for HellaSwag).

## Docker build command
docker build --network host -t gpt-container:gpt .

---

## Block Influence Measurement (Weeks 5-8)

Measures block/layer-level redundancy in GPT-2, per the group proposal's Section 6
("Measurement plan"), reusing `models.load_model`, `data.load_wikitext`, and
`data.prepare_encodings` from the baseline pipeline.

```
run_measurement.py
  │
  ├─ utils.build_measurement_config()      → merge JSON config + CLI args
  ├─ models.load_model()                   → GPT-2 on best device, float32 by default
  ├─ data.load_wikitext() / prepare_encodings()  → WikiText-2 sliding-window calibration set
  ├─ metrics.collect_layer_stats()         → BI scores, relative residual updates, per-layer mean vectors
  ├─ metrics.compute_cosine_similarity_matrix()  → (L+1)x(L+1) centered layer-pair similarity
  ├─ utils.build_measurement_result_dict() → structured results dict
  ├─ utils.save_measurement_results()      → experiments/measurement_gpt2_<timestamp>.json
  ├─ utils.print_measurement_summary()     → most/least redundant block to stdout
  └─ plotting.plot_bi_bar / plot_similarity_heatmap / plot_residual_norms
       → reports/group1/figures/{bi_bar,cosine_heatmap,residual_norms}.png
```

### `src/redundancy/metrics/block_influence.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `collect_layer_stats` | `(model, input_windows, device, max_windows) → LayerStats` | Single forward pass per calibration window with `output_hidden_states=True`. Accumulates running per-layer sums (never stores the full hidden-state stack). Keeps **two separate accumulators**: summed per-token input/output cosines for BI, and running per-layer mean vectors for the similarity matrix. |
| `compute_cosine_similarity_matrix` | `(layer_means, center=True) → np.ndarray` | `(L+1)x(L+1)` cosine similarity between per-layer representations from `LayerStats.layer_means`. Centers the layer means by default (see note below). Reveals similarity structure beyond adjacent layers. |

**Block Influence formula:** `BI_i = 1 - mean_t cos_sim(X_i,t, X_{i+1,t})`, where `X_i,t` is the
hidden state of token `t` entering block `i`. The cosine is computed **per token, then averaged**
— *not* as the cosine of token-averaged vectors, since `cos(mean a, mean b) ≠ mean cos(a, b)`.
GPT-2's forward pass already returns each block's input/output as consecutive entries of
`hidden_states` when `output_hidden_states=True`, so no custom hooks are required. Low BI means
the block's output is nearly identical to its input, i.e. the block is redundant.

**Relative residual update (cross-check):** reported as `mean_t (‖X_{i+1,t} - X_i,t‖ / ‖X_i,t‖)`,
*not* the raw update norm. The residual-stream magnitude grows with depth regardless of
redundancy, so the raw norm isn't comparable block-to-block; normalizing by the input magnitude
makes it a fair "how much did this block change things, in proportion to the signal" measure.

**Heatmap centering / saturation artifact:** the GPT-2 residual stream shares a large common
direction (plus known outlier dimensions), so a *raw* cosine between layer means saturates near
1.0 almost everywhere and washes out — an artifact, not evidence that all layers are redundant.
`compute_cosine_similarity_matrix` therefore centers the layer means (subtracts their across-layer
mean) before taking the cosine, a cheap stand-in for linear CKA (which centers for the same
reason). CKA proper remains the documented follow-up.

**Why float32 for measurement?** `load_model` defaults to float16 on GPU/MPS for speed, but the
proposal flags float16 as a source of numerical noise for similarity-based measurement. The
measurement config (`configs/measurement.json`) sets `"dtype": "float32"` to avoid this.

### `src/redundancy/plotting.py`

| Function | Description |
|----------|-------------|
| `plot_bi_bar` | Per-layer BI bar chart. |
| `plot_similarity_heatmap` | Reused for the cosine similarity heatmap (and any future CKA heatmap). |
| `plot_residual_norms` | Per-layer relative residual-update bar chart, the cross-check metric. |

### `src/utils/__init__.py` additions

| Function | Description |
|----------|-------------|
| `build_measurement_config` | Merges `configs/measurement.json` with CLI overrides (model name, max length, stride, dtype, seed, device, max-windows). |
| `build_measurement_result_dict` | Structured results dict: model, dataset, metric name, seed, device, dtype, full reproduction command, BI scores, relative residual updates, similarity matrix. |
| `save_measurement_results` | Writes to `experiments/measurement_<model>_<timestamp>.json`. |
| `print_measurement_summary` | Prints the most- and least-redundant block indices. |

### Configuration (`configs/measurement.json`)

| Key | Default | Description |
|-----|---------|-------------|
| `model_name` | `"gpt2"` | HuggingFace model identifier |
| `dataset` | `"wikitext-2-raw-v1"` | WikiText-2 raw subset |
| `max_length` | `1024` | Context window size (tokens) |
| `stride` | `1024` | Sliding-window step. Set equal to `max_length` (non-overlapping) for measurement so no token is counted twice in the BI average — unlike the baseline perplexity run, which overlaps windows (stride 512) for context. |
| `dtype` | `"float32"` | Forward-pass precision (float32 to avoid float16 noise in similarity scores) |
| `seed` | `42` | Random seed |

`--max-windows N` subsamples the calibration set for a fast smoke test; the committed
`experiments/measurement_*.json` and `reports/group1/figures/*.png` were produced with the
full (non-subsampled) WikiText-2 test-split calibration set.

### Scope (Weeks 5-8 vs. later weeks)

This pass covers measurement and visualization only, per the shared timeline. Explicitly
deferred:
- **GPT-2 medium / model-size comparison** — Weeks 11-12 ("Model" dimension).
- **Attention/FFN sub-layer decomposition via hooks** — optional stretch noted as a limitation
  in the proposal; block-level hidden states cannot separate attention from FFN without it.
- **Linear CKA heatmap** — the proposal mentions "CKA / cosine heatmap"; cosine alone satisfies
  the Weeks 5-8 bar, CKA is a documented follow-on.
- **Pruning / block removal** — Weeks 9-10.