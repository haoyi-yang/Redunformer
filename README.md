# Redunformer

**Project Seminar: Redundancy in Large Language Models — SS 2026**
**Group 9: Block / Layer-level Redundancy**

## Setup

```bash
# Requires Python >= 3.12 and uv
uv sync
```

## Running the Baseline

```bash
# Perplexity only (fast)
uv run python scripts/run_baseline.py --skip-lm-eval

# Full baseline (perplexity + HellaSwag via lm-eval-harness)
uv run python scripts/run_baseline.py --config configs/baseline.json
```

Results are saved as JSON in `experiments/`.

## Running the Block Influence Measurement (Weeks 5-8)

Measures per-block redundancy with the Block Influence (BI) score from ShortGPT
(Men et al., 2024): `BI_i = 1 - mean_t cos_sim(X_i,t, X_{i+1,t})`, computed directly from
GPT-2's `output_hidden_states=True` output (no custom hooks needed). Also produces a relative
residual-update cross-check and an (L+1)x(L+1) cosine similarity heatmap across all
layer pairs.

```bash
# Full calibration set (all WikiText-2 test windows)
uv run python scripts/run_measurement.py --config configs/measurement.json

# Quick smoke test on a handful of windows
uv run python scripts/run_measurement.py --config configs/measurement.json --max-windows 20
```

Results are saved as JSON in `experiments/`; figures are saved (and committed) to
`reports/group9/figures/`: `bi_bar.png`, `cosine_heatmap.png`, `residual_norms.png`.

## Structure

```
├── configs/
│   ├── baseline.json            # Default baseline-evaluation config
│   └── measurement.json         # Default Block Influence measurement config
├── experiments/                  # Result JSONs
├── reports/group9/figures/       # Committed measurement plots
├── scripts/
│   ├── run_baseline.py          # Weeks 1-2 baseline entry point
│   └── run_measurement.py       # Weeks 5-8 Block Influence measurement entry point
└── src/
    ├── redundancy/
    │   ├── models.py            # Load GPT-2 + auto device selection
    │   ├── data.py              # Load WikiText-2 + sliding-window tokenisation
    │   ├── eval.py              # Perplexity computation + lm-eval wrapper
    │   ├── metrics/
    │   │   └── block_influence.py  # BI score, residual norms, layer similarity matrix
    │   └── plotting.py          # BI bar chart, similarity heatmap, residual-norm plot
    └── utils/                   # Helper functions (config, saving, printing)
```

## Model & Dataset

- **Model**: GPT-2 (124M params, 12 layers)
- **Dataset**: WikiText-2 (raw, test split)
- **Metrics**: Perplexity (sliding-window) + HellaSwag (lm-eval-harness) + Block Influence
