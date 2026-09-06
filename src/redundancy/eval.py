"""Evaluation: perplexity and lm-evaluation-harness wrapper."""

import torch
from torch.nn import CrossEntropyLoss


@torch.no_grad()
def compute_perplexity(
    model,
    input_ids: torch.Tensor,
    device,
    stride: int = 512,
) -> float:
    """Sliding-window perplexity.

    Only scores non-overlapping tokens to avoid double-counting.
    """
    max_length = input_ids.size(1)
    n_windows = input_ids.size(0)

    loss_fn = CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    total_tokens = 0

    for i in range(n_windows):
        ids = input_ids[i : i + 1].to(device)
        logits = model(ids, labels=ids).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()

        start = 0 if i == 0 else max_length - stride - 1

        losses = loss_fn(
            shift_logits[:, start:, :].reshape(-1, shift_logits.size(-1)),
            shift_labels[:, start:].reshape(-1),
        )

        total_loss += losses.sum().item()
        total_tokens += losses.numel()

        if (i + 1) % 50 == 0 or (i + 1) == n_windows:
            ppl = torch.exp(
                torch.tensor(total_loss / total_tokens)
            ).item()
            print(
                f"  [{i + 1}/{n_windows}] running ppl = {ppl:.2f}"
            )

    return torch.exp(
        torch.tensor(total_loss / total_tokens)
    ).item()


def run_lm_eval(
    model_name: str,
    tasks: list[str],
    device: str | None = None,
    batch_size: int | str = "auto",
    model=None,
    tokenizer=None,
):
    """Run lm-evaluation-harness on an already-loaded HF model.

    Passing the existing model is important for pruning experiments:
    lm-eval must evaluate the actual pruned model rather than loading
    a fresh copy of the original model.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    print(
        f"Running lm-eval: model='{model_name}', "
        f"tasks={tasks} ..."
    )

    if model is None or tokenizer is None:
        raise ValueError(
            "run_lm_eval requires the already-loaded model and tokenizer."
        )

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=batch_size,
        device=device,
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
    )

    for task, metrics in results.get("results", {}).items():
        for key, value in metrics.items():
            if (
                not key.startswith("alias")
                and isinstance(value, float)
            ):
                print(
                    f"  {task} / {key} = {value:.4f}"
                )

    return results