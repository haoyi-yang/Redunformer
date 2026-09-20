# RTX 5080 local setup

The repository is checked out on branch `Block`.

The project-local environment uses Python 3.12, the dependencies from `uv.lock`,
and PyTorch 2.12.0+cu130 (CUDA 13.0). The NVIDIA driver detected during setup was
616.56. Python and uv are stored under `.tools`; the environment is `.venv`.

Run commands in PowerShell from this directory. Add the virtual environment to
this terminal's PATH and use the original three scripts in order:

```powershell
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
$env:HF_HOME = "$PWD\.cache\huggingface"
$env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
$env:PYTHONUTF8 = '1'
python scripts/run_baseline.py --config configs/ministral_3b.json
python scripts/run_measurement.py --config configs/ministral_3b_measurement.json
python scripts/run_pruned.py --config configs/ministral_3b.json
```

`run-local.ps1` invokes the environment directly and keeps Hugging Face model
downloads and caches in `.cache`. It also enables UTF-8 console output.
Using plain `uv sync` or `uv run` can replace the CUDA-specific PyTorch build
with the build in the upstream lockfile. Use the environment's Python directly
as above, or the optional launcher, for experiments.

To reinstall the environment:

```powershell
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.tools\python"
$env:UV_CACHE_DIR = "$PWD\.cache\uv"
.\.tools\uv\uv.exe sync --python 3.12 --locked --no-dev --no-install-package torch
.\.tools\uv\uv.exe pip install --python .venv/Scripts/python.exe torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
```

## Model sizing

The RTX 5080 has 16 GB VRAM, part of which is used by Windows. Model weights alone
use approximately two bytes per parameter in FP16/BF16; activations, logits, and
attention caches need additional memory. The upstream pruning script deep-copies
the model on the GPU. The local pruning change keeps the reference on CPU and
moves only the pruned copy to GPU, releasing it after each step.

Ministral has been downloaded and passed a two-window GPU measurement smoke test.
Tiny random Mistral and Llama architecture tests also passed GPU pruning,
measurement, and perplexity checks; those tests are not pretrained model results.

The full Ministral and Llama sequences use WikiText-2 and HellaSwag. Pruning automatically
selects the newest measurement for the configured model from `experiments/`.
It preserves the first and last blocks, evaluates removal of every interior block
in lowest-BI order, then runs three random sweeps of 1–5 removals with seeds 42,
43, and 44. Results are checkpointed after every completed step.
Baseline and pruning use FP16; BI measurement uses BF16 with FP32 reductions.

Both model runs completed successfully. Access to `meta-llama/Llama-3.2-1B`
requires an authorized Hugging Face login for anyone reproducing that run.
