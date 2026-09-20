# Ministral 3B and Llama 3.2 1B block-pruning results

Both models were evaluated on an RTX 5080 with PyTorch 2.12.0+cu130. Baseline and
pruning evaluations used FP16. Block Influence (BI) measurement used BF16 with
FP32 reductions. Perplexity was measured on WikiText-2 and normalized accuracy on
the full HellaSwag validation set.

The pruning sweep always preserves the original first and last transformer
blocks. The lowest-BI run progressively removes every interior block in the BI
ranking measured once on the unpruned model. Three deterministic random runs use
seeds 42, 43, and 44 and remove up to five interior blocks.

## Baselines and BI

| Model | Blocks | Baseline perplexity | HellaSwag acc. | Lowest-BI block | BI | Highest-BI block | BI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ministral 3B Instruct | 14 | 46.67 | 39.25% | 9 | 0.0949 | 0 | 0.4288 |
| Llama 3.2 1B | 16 | 9.28 | 64.20% | 12 | 0.1138 | 0 | 0.6376 |

## Lowest-BI pruning

| Model | Removed blocks | k=1 perplexity | k=1 acc. | k=5 perplexity | k=5 acc. | Only endpoints left: perplexity | Only endpoints left: acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| Ministral 3B Instruct | 9, 10, 8, 11, 12, 7, 6, 4, 5, 3, 2, 1 | 49.02 | 36.80% | 312.33 | 30.68% | 126,845.95 | 25.56% |
| Llama 3.2 1B | 12, 13, 14, 11, 10, 9, 7, 6, 8, 5, 4, 2, 3, 1 | 13.57 | 59.51% | 937.53 | 37.89% | 123,906.38 | 26.34% |

## Random pruning controls

| Model | Run | Removal order | k=1 perplexity | k=1 acc. | k=5 perplexity | k=5 acc. |
|---|---|---|---:|---:|---:|---:|
| Ministral 3B Instruct | seed 42 | 11, 2, 1, 5, 4 | 82.57 | 37.13% | 70,300,184.00 | 26.02% |
| Ministral 3B Instruct | seed 43 | 1, 5, 3, 8, 6 | 23,690.84 | 25.70% | 671,784.00 | 25.62% |
| Ministral 3B Instruct | seed 44 | 7, 9, 11, 2, 3 | 50.41 | 36.16% | 557.94 | 27.38% |
| Llama 3.2 1B | seed 42 | 11, 2, 1, 5, 4 | 17.36 | 57.36% | 2,332.35 | 27.64% |
| Llama 3.2 1B | seed 43 | 1, 5, 12, 3, 8 | 1,224.58 | 28.18% | 6,500.30 | 27.10% |
| Llama 3.2 1B | seed 44 | 7, 9, 13, 2, 3 | 12.79 | 54.60% | 2,747.82 | 27.65% |

The random controls have very high variance because some orders remove block 1
immediately. BI ranks the endpoint-adjacent early blocks as influential and delays
their removal. This gives the lowest-BI strategy a much smoother initial pruning
curve, although both models degrade substantially after several blocks are removed.

The timestamped JSON files in `experiments/` contain every evaluated point,
standard errors, removal orders, system versions, and measurement matrices. The
corresponding BI, centered cosine-similarity, and residual-norm figures are in
`reports/group9/figures/`.
