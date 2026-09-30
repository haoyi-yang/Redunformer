# H1 — depth tertiles (from existing measurement artifacts)

H1 predicted that **middle** FFN layers are *less* redundant than early/late.
The measurement proxy is the importance share of the bottom 10% of neurons:
a **smaller** share means those neurons are cheaper to drop (more redundant).
So H1 is supported if the middle tertile's bottom-10% share is **larger** than
early and late (middle less redundant).

| model | tertile | layers | bottom-10% importance share | mean firing freq |
| --- | --- | --- | ---: | ---: |
| gpt2 | early | 0,1,2,3 | 6.29% | 0.9958 |
| gpt2 | middle | 4,5,6,7 | 6.76% | 0.9976 |
| gpt2 | deep | 8,9,10,11 | 6.30% | 0.9969 |
| gpt2 | *source* | `measurement_gpt2_20260814T105354Z.json` | | |
| gpt2-medium | early | 0,1,2,3,4,5,6,7 | 6.55% | 0.9934 |
| gpt2-medium | middle | 8,9,10,11,12,13,14,15 | 6.93% | 0.9978 |
| gpt2-medium | deep | 16,17,18,19,20,21,22,23 | 6.29% | 0.9936 |
| gpt2-medium | *source* | `measurement_gpt2-medium_20260814T115747Z.json` | | |
| gpt2-large | early | 0,1,2,3,4,5,6,7,8,9,10,11 | 6.41% | 0.9821 |
| gpt2-large | middle | 12,13,14,15,16,17,18,19,20,21,22,23 | 7.09% | 0.9955 |
| gpt2-large | deep | 24,25,26,27,28,29,30,31,32,33,34,35 | 6.23% | 0.9885 |
| gpt2-large | *source* | `measurement_gpt2-large_20260814T121109Z.json` | | |
| Llama-2-7B | early | 0,1,2,3,4,5,6,7,8,9 | 6.67% | 0.9665 |
| Llama-2-7B | middle | 10,11,12,13,14,15,16,17,18,19 | 6.93% | 0.9950 |
| Llama-2-7B | deep | 20,21,22,23,24,25,26,27,28,29,30,31 | 6.81% | 0.9928 |
| Llama-2-7B | *source* | `measurement_meta-llama__Llama-2-7b-hf_20260814T131909Z.json` | | |
| Qwen3-0.6B | early | 0,1,2,3,4,5,6,7,8 | 5.87% | 0.9894 |
| Qwen3-0.6B | middle | 9,10,11,12,13,14,15,16,17 | 4.87% | 0.9943 |
| Qwen3-0.6B | deep | 18,19,20,21,22,23,24,25,26,27 | 3.15% | 0.9841 |
| Qwen3-0.6B | *source* | `measurement_Qwen__Qwen3-0.6B_20260629T182308Z.json` | | |
| Qwen3.5-4B | early | 0,1,2,3,4,5,6,7,8,9 | 7.19% | 0.9950 |
| Qwen3.5-4B | middle | 10,11,12,13,14,15,16,17,18,19 | 6.38% | 0.9958 |
| Qwen3.5-4B | deep | 20,21,22,23,24,25,26,27,28,29,30,31 | 5.73% | 0.9965 |
| Qwen3.5-4B | *source* | `measurement_Qwen__Qwen3.5-4B_20260629T183140Z.json` | | |

## Ablation (10% lowest-importance neurons, tertile layers only)

The static share above is only a *proxy*. This section masks 10% of the
neurons in each tertile and measures the actual cost, which is what H1
is really about. H1 predicted the **middle** tertile would hurt most.

| model | early ΔPPL | middle ΔPPL | deep ΔPPL | most expensive tertile |
| --- | ---: | ---: | ---: | --- |
| gpt2 | +5.24 | +4.08 | +2.31 | early |
| gpt2-medium | +7.06 | +2.92 | +1.77 | early |
| gpt2-large | +1.06 | +0.98 | +1.01 | flat |
| Llama-2-7B | +0.27 | +0.10 | +0.42 | deep |
| Qwen3-0.6B | +2.69 | +0.19 | +0.81 | early |
| Qwen3.5-4B | +0.34 | +0.37 | +1.02 | deep |

Eval slice matches each model's own masking config, so ΔPPL is comparable
within a row but not across rows (baselines differ by up to 5×).
