from redundancy.models import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-0.6B")

for name, param in model.named_parameters():
    if len(param.shape) == 2:
        rows, cols = param.shape
        print(f"{rows*cols:>10} | {name:60} | {param.shape}")