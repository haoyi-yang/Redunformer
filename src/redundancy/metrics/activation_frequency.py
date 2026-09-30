from redundancy.config import load_yaml
from redundancy.models import load_causal_lm
from redundancy.hooks import ActivationStore, hook_gate_up_proj

import torch

cfg = load_yaml("configs/models/qwen3-0.6b.yaml")

loaded = load_causal_lm(cfg)

model = loaded.model
tokenizer = loaded.tokenizer

store = ActivationStore()

hook_gate_up_proj(model, store)

inputs = tokenizer(
    "We measure redundancy.",
    return_tensors = "pt"
)

with torch.no_grad():
    model(**inputs)
    
print("Number of hooked layers: ", len(store._buffers))

for name, tensors in store._buffers.items():
    print(name)
    print("Number of forward passes: ", len(tensors))
    print("Shape: ", tensors[0].shape)
    break
store.remove_hooks()
