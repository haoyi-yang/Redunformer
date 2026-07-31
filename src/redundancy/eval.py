import torch
from tqdm import tqdm

def evaluate_perplexity(model, tokenizer, dataset, text_column="text", stride=512, max_length=1024, eval_frac=1.0):
    device = model.device

    # Filter out empty strings if any
    texts = [t for t in dataset[text_column] if t.strip()]
    full_text = "\n\n".join(texts)

    print("Tokenizing evaluation dataset...")
    encodings = tokenizer(full_text, return_tensors="pt")

    seq_len = encodings.input_ids.size(1)
    # For fast screening a subset (the first eval_frac of the corpus) may be used.
    # The same fraction is applied to every comparable run, so rankings stay valid.
    eval_len = seq_len if eval_frac >= 1.0 else max(max_length, int(seq_len * eval_frac))
    print(f"Total tokens for evaluation: {eval_len} of {seq_len}")

    nlls = []
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, eval_len, stride), desc="Evaluating perplexity"):
        end_loc = min(begin_loc + max_length, eval_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == eval_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


# ---------------------------------------------------------------------------
# Output-distribution divergence (dense vs. pruned)
#
# Perplexity alone can hide task-level degradation, so we also measure how far
# the pruned model's behaviour drifts from the dense model on a small fixed
# probe: the KL divergence of the output distributions, the top-1 next-token
# agreement rate, and the cosine similarity of the final hidden states.
# ---------------------------------------------------------------------------

def build_divergence_probe(tokenizer, dataset, n_seqs=4, seqlen=128, text_column="text"):
    """A small fixed set of token windows [n_seqs, seqlen] for divergence probing."""
    texts = [t for t in dataset[text_column] if t.strip()]
    ids = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]

    chunks = []
    for i in range(n_seqs):
        start = i * seqlen
        if start + seqlen <= ids.size(0):
            chunks.append(ids[start:start + seqlen])

    if not chunks:
        raise ValueError("Not enough tokens to build the divergence probe.")

    return torch.stack(chunks)


@torch.no_grad()
def dense_reference_outputs(model, probe_ids):
    """Run the (dense) model on the probe and cache what the divergence needs.

    Computed once on the fully dense model and reused for every experiment.
    Stored on CPU in half precision to keep memory small.
    """
    device = model.device
    out = model(probe_ids.to(device), output_hidden_states=True)

    logits = out.logits.reshape(-1, out.logits.size(-1)).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    hidden = out.hidden_states[-1].reshape(-1, out.hidden_states[-1].size(-1)).float()

    return {
        "logprobs": logprobs.half().cpu(),
        "argmax": logprobs.argmax(-1).cpu(),
        "hidden": hidden.half().cpu(),
    }


@torch.no_grad()
def output_divergence(model, probe_ids, reference):
    """Compare the current (pruned) model's outputs to the dense reference."""
    device = model.device
    out = model(probe_ids.to(device), output_hidden_states=True)

    logits = out.logits.reshape(-1, out.logits.size(-1)).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    hidden = out.hidden_states[-1].reshape(-1, out.hidden_states[-1].size(-1)).float()

    ref_logprobs = reference["logprobs"].float().to(device)
    ref_argmax = reference["argmax"].to(device)
    ref_hidden = reference["hidden"].float().to(device)

    # KL(dense || pruned) = sum_i P_dense(i) * (log P_dense(i) - log P_pruned(i))
    p_dense = ref_logprobs.exp()
    kl = (p_dense * (ref_logprobs - logprobs)).sum(-1).mean().item()

    top1 = (logprobs.argmax(-1) == ref_argmax).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(hidden, ref_hidden, dim=-1).mean().item()

    return {"kl_dense_pruned": kl, "top1_agreement": top1, "hidden_cosine": cos}
