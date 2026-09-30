# Weeks 1–2: Baseline evaluation note (Group 5 — Neuron)

## Model

| Field | Value |
|-------|--------|
| **Target model** | `Qwen/Qwen3.5-4B` (4-bit NF4, ~8 GB VRAM) |
| **Weights location** | `models/cache/Qwen3.5-4B/` (local download, gitignored) |
| **Pipeline verification (earlier)** | `Qwen/Qwen3-0.6B` (4-bit fallback) |

**Stack:** `torch 2.11.0+cu128`, `transformers 5.10.2`

**Justification:** Qwen3.5 is the seminar’s suggested family. 4-bit loading fits the RTX 4060 8 GB GPU. Neuron/FFN redundancy work will target feed-forward blocks inside Qwen3.5 hybrid layers.

## Dataset

| Field | Value |
|-------|--------|
| Dataset | WikiText-2 (`wikitext`, config `wikitext-2-raw-v1`) |
| Split | `test` |

## Evaluation commands

**Download weights (uses `HF_TOKEN` from `.env`):**

```powershell
cd Redunformer-Neuron
$env:Path = "C:\Users\gencg\.local\bin;$env:Path"
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*([^#][^=]*?)=(.*)$') {
    Set-Item -Path "env:$($matches[1].Trim())" -Value $matches[2].Trim().Trim('"').Trim("'")
  }
}
uv run python scripts/download_model.py --method resumable
```

**Full test split baseline (Qwen3.5-4B, 4-bit, all 2,891 docs):**

```powershell
uv run python scripts/run_baseline.py --config configs/eval/baseline_full.yaml
```

**Dev subset (32 docs):**

```powershell
uv run python scripts/run_baseline.py --config configs/eval/baseline.yaml
```

**Smoke test (8 docs):**

```powershell
uv run python scripts/run_baseline.py --config configs/eval/baseline_qwen35_4b_smoke.yaml
```

## Baseline results

| Model | Subset | WikiText-2 perplexity | Tokens | Result file |
|-------|--------|----------------------|--------|-------------|
| **`Qwen/Qwen3.5-4B` (4-bit)** | **full test (2,891 docs)** | **17.06** | **297,053** | `baseline_Qwen__Qwen3.5-4B_20260605T172429Z.json` |
| **`Qwen/Qwen3.5-9B` (4-bit)** | **full test** | **12.90** | — | `reports/group_5/artifacts/baseline_Qwen__Qwen3.5-9B_20260620T075843Z.json` (from #11; Henrik, 2026-06-20) |
| `Qwen/Qwen3.5-4B` (4-bit) | 32 docs (14 non-empty) | 19.16 | 1,355 | `baseline_Qwen__Qwen3.5-4B_20260605T134607Z.json` |
| `Qwen/Qwen3.5-4B` (4-bit) | 8 docs (smoke) | 16.25 | 428 | `baseline_Qwen__Qwen3.5-4B_20260605T134508Z.json` |
| `Qwen/Qwen3-0.6B` (4-bit, fallback) | full test | 39.78 | 298,938 | `baseline_Qwen__Qwen3-0.6B_20260604T232230Z.json` |
| **`gpt2` (fp32)** | **full test (2,891 docs)** | **50.07** | **283,287** | `baseline_gpt2_20260823T075140Z.json` |
| **`gpt2-medium` (fp32)** | **full test** | **36.65** | **283,287** | `baseline_gpt2-medium_20260823T075340Z.json` |
| **`gpt2-large` (fp32)** | **full test** | **31.99** | **283,287** | `baseline_gpt2-large_20260823T075700Z.json` |
| **`meta-llama/Llama-2-7b-hf` (4-bit)** | **full test** | **10.58** | **335,644** | `baseline_meta-llama__Llama-2-7b-hf_20260823T080902Z.json` |

GPT-2 and Llama-2-7B full-split baselines were added 2026-08-23 (same WikiText-2 test split, same harness). GPT-2 uses a 1024-token window; Llama is 4-bit NF4. Those numbers are therefore not comparable to Qwen as raw PPL — only as each model's own full-split reference.

**9B reproduce command** (needs ~16 GB GPU):

```powershell
uv run python scripts/run_baseline.py --config configs/eval/baseline_qwen3.5-9b.yaml
```

GPT-2 / Llama-2 (full test, same harness):

```powershell
uv run python scripts/run_baseline.py --config configs/eval/baseline_gpt2.yaml
uv run python scripts/run_baseline.py --config configs/eval/baseline_gpt2-medium.yaml
uv run python scripts/run_baseline.py --config configs/eval/baseline_gpt2-large.yaml
uv run python scripts/run_baseline.py --config configs/eval/baseline_llama2-7b.yaml
```

Model config stub: `configs/models/qwen3.5-9b.yaml`. Hardware from #11: 7.35 GiB allocated / 15.99 GiB total.

## Hardware

| Field | Value |
|-------|--------|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8 GB |
| PyTorch | `2.11.0+cu128` |

## Status

- [x] Repository scaffold and shared `src/redundancy` package
- [x] uv / `pyproject.toml` + CUDA PyTorch cu128 + transformers 5.x
- [x] WikiText-2 loader verified
- [x] Qwen3.5-4B weights downloaded locally
- [x] GPU baseline pipeline run (Qwen3.5-4B, 4-bit)
- [x] Perplexity recorded in JSON + this note
- [x] Code pushed to internal GitHub repo
- [x] Full WikiText-2 test split (2,891 docs, PPL **17.06**)
- [x] Same full-split baseline for GPT-2 124M/355M/774M and Llama-2-7B (2026-08-23)
- [ ] Open PR to shared seminar repo

## Reproducibility

```json
"seed": 42,
"command": "scripts/run_baseline.py --config configs/eval/baseline_full.yaml",
"metrics.perplexity.perplexity": 17.056259177791453
```
