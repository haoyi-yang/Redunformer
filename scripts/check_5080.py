"""GPU infrastructure checks with tiny random models; no pretrained downloads."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from transformers import MistralConfig, MistralForCausalLM, LlamaConfig, LlamaForCausalLM
from redundancy.models import clone_and_prune_model
from redundancy.metrics import collect_layer_stats
from redundancy.eval import compute_perplexity
from run_pruned import select_blocks

assert torch.cuda.is_available()
assert select_blocks([-9, 3, 1, 2, -8], 3) == [2, 3, 1]
ids = torch.randint(0, 128, (2, 32))
for config_cls, model_cls in [(MistralConfig, MistralForCausalLM), (LlamaConfig, LlamaForCausalLM)]:
    config = config_cls(vocab_size=128, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=7, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16)
    model = model_cls(config).to(dtype=torch.bfloat16).eval()
    pruned = clone_and_prune_model(model, [1, 2, 3, 4, 5]).to("cuda")
    assert len(pruned.model.layers) == 2 and len(model.model.layers) == 7
    assert next(model.parameters()).device.type == "cpu"
    for original_index, kept in [(0, 0), (6, 1)]:
        assert torch.equal(model.model.layers[original_index].self_attn.q_proj.weight,
                           pruned.model.layers[kept].self_attn.q_proj.weight.cpu())
    assert [b.self_attn.layer_idx for b in pruned.model.layers] == [0, 1]
    stats = collect_layer_stats(pruned, ids, torch.device("cuda"), max_windows=2)
    ppl = compute_perplexity(pruned, ids, torch.device("cuda"), stride=16)
    assert stats.n_layers == 2 and np.isfinite(stats.bi_scores).all()
    assert np.isfinite(ppl) and ppl > 0
    print(model_cls.__name__, "CPU reference / GPU pruning + BI + perplexity: PASS", flush=True)
    del pruned, model
    torch.cuda.empty_cache()
print(torch.cuda.get_device_name(0), torch.__version__, "PASS")
