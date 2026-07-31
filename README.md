# Redunformer

Cross-granularity redundancy analysis for large language models (SoSe 2026).

**Group:** Weight-level redundancy  
**Branch:** `Weight`

## Quick start

Requires **NVIDIA GPU** with Docker GPU support (`--gpus all`) or Podman on mlsp (`--device nvidia.com/gpu=all`).

**Driver note:** the container ships `torch 2.12+cu130`. Host NVIDIA driver must support **CUDA 13.x** (e.g. driver **610+** on Windows). Older drivers fail with `NVIDIA driver too old`.

**Windows:** install [GNU Make](https://strawberryperl.com/) (e.g. Strawberry Perl → `C:\Strawberry\c\bin` on PATH) and Docker Desktop with GPU support.

**Validated locally (Weight branch):** `make verify`, `make smoke`, `make qwen-small` on RTX 3060 Laptop — see [baseline runs report](reports/weight_level/baseline_runs_2026-06-06.md).

### 1. Build the container

```bash
make build
# on mlsp student pool:
make CONTAINER=podman build
```

### 2. Verify setup (fast, no model download)

```bash
make verify
```

Checks imports and configs inside the GPU container. Downloads nothing from Hugging Face.

### 3. Run baseline evaluations (GPU)

```bash
make smoke              # gpt2, limit=0.01 by default
make smoke LIMIT=0.05
make qwen-small         # Qwen3-1.7B (intermediate run on tight VRAM)
make qwen               # Qwen3-4B — main baseline
```

On mlsp:

```bash
make CONTAINER=podman verify
make CONTAINER=podman qwen
```

### 4. Run pruning pipeline (GPU)

The pruning pipeline provides an interactive workflow for:

1. Downloading a model from Hugging Face
2. Applying a pruning algorithm
3. Saving the pruned model
4. Running lm-eval-harness evaluation
5. Saving evaluation results

```bash
make pipeline
```

On mlsp:

```bash
make CONTAINER=podman pipeline
```

**Interactive** (`make pipeline`): prompts for model, algorithm, and parameters.

**Non-interactive** — pass Make variables, no questions:

```bash
make pipeline ALGORITHM=sparsegpt MODEL=gpt2 SPARSITY=0.5 NSAMPLES=32 SEQLEN=512 SKIP_EVAL=1
```

The resulting model is stored under:

	experiments/pruned/

Evaluation results are stored under:

	experiments/baseline/


### Pruning algorithm API

All pruning algorithms must be placed in:

```text
src/redundancy/pruning/
```

Each algorithm is implemented as a standalone Python module.

Required interface:

```text
ALGORITHM_NAME = "random_pruning"

PARAMETERS = {
    "sparsity": {
        "type": float,
        "prompt": "Sparsity (0-1)"
    }
}

def prune(model, **kwargs):
    ...
```

## Required objects

| Object | Purpose |
|--------|---------|
| `ALGORITHM_NAME` | Human-readable algorithm name |
| `PARAMETERS` | Parameters requested from the user |
| `prune()` | Applies pruning to the model |

### Adding a new algorithm

Create a new file:

```bash
src/redundancy/pruning/my_algorithm.py
```

Implement the required interface.

The pruning pipeline automatically discovers available algorithms and presents them in the selection menu. 
No modifications to run_pruning_pipeline.py are required.

### SparseGPT (weight-level, one-shot)

Implements [SparseGPT](https://arxiv.org/abs/2301.00774): layer-wise OBS reconstruction with adaptive mask selection. Supports unstructured sparsity and N:M patterns (e.g. 2:4).

Use via the main pipeline (preferred):

```bash
make pipeline ALGORITHM=sparsegpt MODEL=gpt2 SPARSITY=0.5 NSAMPLES=32 SEQLEN=512 SKIP_EVAL=1
make pipeline ALGORITHM=sparsegpt MODEL=Qwen/Qwen3-1.7B SPARSITY=0.5
```

Optional direct script: `scripts/run_sparsegpt.py`. Module: `src/redundancy/pruning/sparsegpt.py`.

### 5. Run without Make

```bash
docker run --rm --gpus all \
  -v "$(pwd)/experiments:/app/experiments" \
  -v "$(pwd)/configs:/app/configs" \
  -v redunformer_hf_cache:/root/.cache/huggingface \
  redunformer \
  python scripts/run_baseline.py --config configs/models/gpt2.yaml --limit 0.05
```

## Model configs

Model choice is a YAML file under `configs/models/`. The Makefile mounts `configs/` into the container so YAML edits apply without `make build`.

| Config | Model | Use case |
|--------|-------|----------|
| `gpt2.yaml` | `gpt2` | Smoke test |
| `qwen3-1.7b.yaml` | `Qwen/Qwen3-1.7B` | Light GPU baseline (6 GB VRAM) |
| `qwen3-4b.yaml` | `Qwen/Qwen3-4B` | **Main baseline** |

Hugging Face repo names for Qwen3 do not use an `-Instruct` suffix.

Change `pretrained`, `tasks`, `seed`, and `dtype` in the YAML. Weights are downloaded from Hugging Face on first run and cached in the `redunformer_hf_cache` volume.

Recommended: **8GB+ VRAM** for Qwen3-4B (`make qwen`). RTX 3060 Laptop (6 GB) — use `make qwen-small` to validate the pipeline, then run `make qwen` on a machine with enough VRAM or mlsp.

## Baseline results so far

Reference metrics before pruning (0-shot, lm-eval, seed=42). Full tables: [baseline_runs_2026-06-06.md](reports/weight_level/baseline_runs_2026-06-06.md).

| Run | Model | Tasks | Key metrics |
|-----|-------|-------|-------------|
| Smoke | `gpt2` | hellaswag (1%) | acc_norm 0.436 |
| Interim | `Qwen/Qwen3-1.7B` | hellaswag, piqa | hellaswag acc_norm 0.604, piqa acc 0.726 |
| **Main baseline** | `Qwen/Qwen3-4B` | hellaswag, piqa, arc_easy | hellaswag acc_norm **0.684**, piqa acc **0.749**, arc_easy acc **0.806** (mlsp4) |

## Weeks 1–2 checklist

| Done | Task |
|------|------|
| yes | Repo scaffold, Docker + uv, lm-eval wiring |
| yes | `make verify`, `make smoke` (gpt2 on GPU) |
| yes | `make qwen-small` (Qwen3-1.7B interim baseline) |
| yes | Baseline note with metrics in `reports/weight_level/` |
| yes | **`make qwen` — Qwen3-4B main baseline (mlsp4)** |

## Project layout

```text
configs/models/          model + eval configs (mounted into container)
src/redundancy/          shared library (models, eval, plotting)
scripts/                 CLI entrypoints
experiments/baseline/    JSON results (gitignored)
reports/weight_level/    group notes and deliverables
```

## Reproducibility

Each run writes:

- `experiments/baseline/<model>.json` from lm-eval
- `experiments/baseline/<model>.meta.json` with config path, command, seed, tasks, timestamp

Raw JSON stays gitignored (lm-eval outputs can be large; many runs ahead). Reproducibility comes from config + command + seed — re-run `make qwen` to regenerate. Summary metrics live in `reports/weight_level/`.

## Portable setup

On a GPU machine (local 3060+ or mlsp pool):

1. `git clone` + `cd Redunformer` + `git checkout Weight`
2. `make build`
3. `make verify`
4. `make qwen` (or `make smoke` first)

Edit code on any machine; run `make` targets only where NVIDIA + container GPU support works.

On mlsp: `make CONTAINER=podman ...`, SSH with your pool username, RTPT, mlstudentpool Mattermost channel.

## Development with uv (optional, outside Docker)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python scripts/run_baseline.py --config configs/models/gpt2.yaml --limit 0.05
```
