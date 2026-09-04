IMAGE ?= redunformer
CONTAINER ?= docker
PROJECT_DIR := $(CURDIR)
LIMIT ?= 0.01

ifeq ($(CONTAINER),podman)
GPU_FLAGS := --device nvidia.com/gpu=all
else
GPU_FLAGS := --gpus all
endif

RUN_FLAGS := --rm $(GPU_FLAGS) \
	-v $(PROJECT_DIR)/experiments:/app/experiments \
	-v $(PROJECT_DIR)/configs:/app/configs \
	-v redunformer_hf_cache:/root/.cache/huggingface

# Pipeline knobs (set ALGORITHM=... for non-interactive run, no prompts)
MODEL ?= gpt2
ALGORITHM ?=
SPARSITY ?= 0.5
NSAMPLES ?= 128
SEQLEN ?= 2048
BLOCKSIZE ?= 128
PERCDAMP ?= 0.01
DATASET ?= wikitext2
SEED ?= 42
PRUNEN ?= 0
PRUNEM ?= 0
BASE ?= wanda
CYCLES ?= 50
EPSILON ?= 0.1
VAR_POWER ?= 1.0
SAME_SIGN ?= 0
SKIP_LAYER ?= none
DENSE ?=
SKIP_EVAL ?= 0
# CLASSIC=1 keeps original magnitude / sparsegpt (uneven masks, not DSnoT-safe).
CLASSIC ?= 0

.PHONY: help build verify smoke smoke-fast qwen-small qwen shell pipeline sparsegpt

help:
	@echo "Targets (all eval runs use GPU):"
	@echo "  make build         Build container image"
	@echo "  make verify        Fast check (imports + configs, no model download)"
	@echo "  make smoke         gpt2 smoke test on GPU (limit=$(LIMIT))"
	@echo "  make smoke-fast    Alias for smoke"
	@echo "  make qwen-small    Qwen3-1.7B baseline on GPU"
	@echo "  make qwen          Qwen3-4B baseline on GPU"
	@echo "  make pipeline      Pruning + eval (interactive OR pass ALGORITHM=...)"
	@echo "  make shell         Interactive shell in container with GPU"
	@echo ""
	@echo "Examples:"
	@echo "  make verify"
	@echo "  make smoke LIMIT=0.05"
	@echo "  make CONTAINER=podman qwen"
	@echo "  make build && make pipeline"
	@echo "  make pipeline ALGORITHM=sparsegpt MODEL=gpt2 SPARSITY=0.5 NSAMPLES=32 SEQLEN=512 SKIP_EVAL=1"
	@echo "    → auto-uses sparse_dsnot (per-row). Classic block SparseGPT: CLASSIC=1"
	@echo "  make pipeline ALGORITHM=magnitude MODEL=gpt2 SPARSITY=0.5 SKIP_EVAL=1"
	@echo "    → auto-uses magnitude_dsnot (per-row). Classic global magnitude: CLASSIC=1"
	@echo "  make pipeline ALGORITHM=dsnot MODEL=gpt2 SPARSITY=0.5 BASE=wanda SKIP_EVAL=1"
	@echo "  make pipeline ALGORITHM=dsnot MODEL=gpt2 SPARSITY=0.5 BASE=magnitude SKIP_EVAL=1"
	@echo "  make pipeline ALGORITHM=dsnot MODEL=gpt2 SPARSITY=0.5 BASE=sparsegpt SKIP_EVAL=1"
	@echo "  make pipeline ALGORITHM=dsnot MODEL=experiments/pruned/gpt2-wanda50 DENSE=gpt2 SKIP_EVAL=1"
	@echo "  make CONTAINER=podman build && make CONTAINER=podman pipeline"

build:
	$(CONTAINER) build -t $(IMAGE) .

verify:
	$(CONTAINER) run $(RUN_FLAGS) $(IMAGE) python scripts/verify_setup.py

smoke smoke-fast:
	$(CONTAINER) run $(RUN_FLAGS) $(IMAGE) \
		python scripts/run_baseline.py --config configs/models/gpt2.yaml --limit $(LIMIT)

qwen-small:
	$(CONTAINER) run $(RUN_FLAGS) $(IMAGE) \
		python scripts/run_baseline.py --config configs/models/qwen3-1.7b.yaml

qwen:
	$(CONTAINER) run $(RUN_FLAGS) $(IMAGE) \
		python scripts/run_baseline.py --config configs/models/qwen3-4b.yaml

shell:
	$(CONTAINER) run -it $(RUN_FLAGS) --entrypoint bash $(IMAGE)

# ---------------------------------------------------------------------------
# Resolve ALGORITHM / BASE so DSnoT always gets per-row-compatible masks.
# ---------------------------------------------------------------------------
PIPELINE_ALGORITHM := $(ALGORITHM)
PIPELINE_BASE := $(BASE)
BASE_LC := $(shell echo '$(BASE)' | tr '[:upper:]' '[:lower:]')

# ALGORITHM=dsnot: accept magnitude_dsnot / sparse_dsnot as BASE aliases.
ifeq ($(strip $(ALGORITHM)),dsnot)
  ifneq ($(filter $(BASE_LC),magnitude magnitude_dsnot),)
    PIPELINE_BASE := magnitude
  endif
  ifneq ($(filter $(BASE_LC),sparsegpt sparse_dsnot),)
    PIPELINE_BASE := sparsegpt
  endif
endif

# First-stage ALGORITHM=magnitude|sparsegpt → DSnoT-safe modules (unless CLASSIC=1).
ifeq ($(CLASSIC),0)
  ifeq ($(strip $(ALGORITHM)),magnitude)
    PIPELINE_ALGORITHM := magnitude_dsnot
  endif
  ifeq ($(strip $(ALGORITHM)),sparsegpt)
    PIPELINE_ALGORITHM := sparse_dsnot
  endif
endif

# Interactive if ALGORITHM is empty; non-interactive when ALGORITHM=... is set.
PIPELINE_CLI :=
ifneq ($(strip $(ALGORITHM)),)
PIPELINE_CLI += --model $(MODEL) --algorithm $(PIPELINE_ALGORITHM)
PIPELINE_CLI += --sparsity $(SPARSITY) --nsamples $(NSAMPLES) --seqlen $(SEQLEN)
PIPELINE_CLI += --blocksize $(BLOCKSIZE) --percdamp $(PERCDAMP)
PIPELINE_CLI += --dataset $(DATASET) --seed $(SEED)
PIPELINE_CLI += --prunen $(PRUNEN) --prunem $(PRUNEM)
PIPELINE_CLI += --base $(PIPELINE_BASE) --cycles $(CYCLES) --epsilon $(EPSILON)
PIPELINE_CLI += --var_power $(VAR_POWER) --same_sign $(SAME_SIGN)
PIPELINE_CLI += --skip_layer $(SKIP_LAYER)
ifneq ($(strip $(DENSE)),)
PIPELINE_CLI += --dense $(DENSE)
endif
ifeq ($(SKIP_EVAL),1)
PIPELINE_CLI += --skip-eval
endif
endif

# Mount only src/scripts so host checkout does not shadow /app/.venv from the image.
PIPELINE_MOUNTS := \
	-v $(PROJECT_DIR)/src:/app/src \
	-v $(PROJECT_DIR)/scripts:/app/scripts

pipeline:
ifeq ($(strip $(ALGORITHM)),)
	$(CONTAINER) run -it $(RUN_FLAGS) $(PIPELINE_MOUNTS) $(IMAGE) \
		python scripts/run_pruning_pipeline.py
else
ifneq ($(PIPELINE_ALGORITHM),$(ALGORITHM))
	@echo "pipeline: ALGORITHM=$(ALGORITHM) → $(PIPELINE_ALGORITHM) (DSnoT-safe; CLASSIC=1 for original)"
endif
ifneq ($(PIPELINE_BASE),$(BASE))
	@echo "pipeline: BASE=$(BASE) → $(PIPELINE_BASE) (DSnoT init alias)"
endif
	$(CONTAINER) run $(RUN_FLAGS) $(PIPELINE_MOUNTS) $(IMAGE) \
		python scripts/run_pruning_pipeline.py $(PIPELINE_CLI)
endif

# Optional thin wrapper around scripts/run_sparsegpt.py (classic block SparseGPT).
sparsegpt:
	$(CONTAINER) run $(RUN_FLAGS) $(PIPELINE_MOUNTS) $(IMAGE) \
		python scripts/run_sparsegpt.py \
			--model $(MODEL) \
			--sparsity $(SPARSITY) \
			--nsamples $(NSAMPLES) \
			--seqlen $(SEQLEN)
