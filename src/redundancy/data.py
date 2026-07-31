import random

from datasets import load_dataset


def load_evaluation_dataset(dataset_name: str = "Salesforce/wikitext", subset: str = "wikitext-2-raw-v1", split: str = "test"):
    print(f"Loading dataset {dataset_name} ({subset}), split: {split}...")
    if dataset_name == "wikitext":
        dataset_name = "Salesforce/wikitext"

    dataset = load_dataset(dataset_name, subset, split=split)
    return dataset


def load_calibration_dataset(tokenizer, n_samples: int = 128, seqlen: int = 512, seed: int = 0,
                             dataset_name: str = "Salesforce/wikitext", subset: str = "wikitext-2-raw-v1"):
    """Return a list of tokenized windows [1, seqlen] for calibration.

    Sampled from the TRAIN split so it stays disjoint from the test split used
    for perplexity. Wanda and SparseGPT both need calibration activations; the
    paper stresses that these statistics depend on the calibration data.
    """
    print(f"Loading calibration data: {n_samples} x {seqlen} tokens from {dataset_name} ({subset}) train...")
    if dataset_name == "wikitext":
        dataset_name = "Salesforce/wikitext"

    data = load_dataset(dataset_name, subset, split="train")
    text = "\n\n".join(t for t in data["text"] if t.strip())
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    total = input_ids.size(1)

    if total <= seqlen:
        raise ValueError(f"Calibration corpus ({total} tokens) shorter than seqlen ({seqlen}).")

    rng = random.Random(seed)
    samples = []
    for _ in range(n_samples):
        start = rng.randint(0, total - seqlen - 1)
        samples.append(input_ids[:, start:start + seqlen])

    return samples
