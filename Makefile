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
SKIP_EVAL ?= 0

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
	@echo "  make pipeline"
	@echo "  make pipeline ALGORITHM=sparsegpt MODEL=gpt2 SPARSITY=0.5 NSAMPLES=32 SEQLEN=512 SKIP_EVAL=1"

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

# Interactive if ALGORITHM is empty; non-interactive when ALGORITHM=... is set.
PIPELINE_CLI :=
ifneq ($(strip $(ALGORITHM)),)
PIPELINE_CLI += --model $(MODEL) --algorithm $(ALGORITHM)
PIPELINE_CLI += --sparsity $(SPARSITY) --nsamples $(NSAMPLES) --seqlen $(SEQLEN)
PIPELINE_CLI += --blocksize $(BLOCKSIZE) --percdamp $(PERCDAMP)
PIPELINE_CLI += --dataset $(DATASET) --seed $(SEED)
PIPELINE_CLI += --prunen $(PRUNEN) --prunem $(PRUNEM)
ifeq ($(SKIP_EVAL),1)
PIPELINE_CLI += --skip-eval
endif
endif

pipeline:
ifeq ($(strip $(ALGORITHM)),)
	podman run -it --rm \
		--security-opt=label=disable \
		--device /dev/nvidia0 \
		--device /dev/nvidiactl \
		--device /dev/nvidia-uvm \
		--device /dev/nvidia-uvm-tools \
		-v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:ro \
		-v $(PWD):/app \
		localhost/redunformer \
		python scripts/run_pruning_pipeline.py
else
	$(CONTAINER) run $(RUN_FLAGS) \
		-v $(PROJECT_DIR)/src:/app/src \
		-v $(PROJECT_DIR)/scripts:/app/scripts \
		$(IMAGE) \
		python scripts/run_pruning_pipeline.py $(PIPELINE_CLI)
endif

# Optional thin wrapper around scripts/run_sparsegpt.py (kept; prefer make pipeline).
sparsegpt:
	$(CONTAINER) run $(RUN_FLAGS) \
		-v $(PROJECT_DIR)/src:/app/src \
		-v $(PROJECT_DIR)/scripts:/app/scripts \
		$(IMAGE) \
		python scripts/run_sparsegpt.py \
			--model $(MODEL) \
			--sparsity $(SPARSITY) \
			--nsamples $(NSAMPLES) \
			--seqlen $(SEQLEN)
