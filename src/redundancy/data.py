"""Dataset loading helpers."""

from __future__ import annotations

from typing import Any

import pyarrow.parquet as pq
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

_WIKITEXT2_SPLITS = ("train", "validation", "test")
_WIKITEXT2_REVISION = "refs/convert/parquet"


def _load_wikitext2_split(split: str) -> Dataset:
    path = hf_hub_download(
        repo_id="wikitext",
        repo_type="dataset",
        filename=f"wikitext-2-raw-v1/{split}/0000.parquet",
        revision=_WIKITEXT2_REVISION,
    )
    table = pq.read_table(path)
    return Dataset(table)


def _load_wikitext2() -> dict[str, Dataset]:
    return {split: _load_wikitext2_split(split) for split in _WIKITEXT2_SPLITS}


def _piqa_prompt(row: dict[str, Any]) -> str:
    """Render a PIQA row the way lm-evaluation-harness scores it.

    Only the *correct* solution is kept, so calibrating on this text measures
    the neurons the task actually relies on rather than those excited by a
    distractor. Matching the harness prompt format matters for H3: the neuron
    ranking and the accuracy metric then refer to the same input distribution.
    """
    solution = row["sol2"] if int(row.get("label", 0)) == 1 else row["sol1"]
    return f"Question: {row['goal']}\nAnswer: {solution}"


# Datasets whose text has to be assembled from several columns.
TEXT_BUILDERS: dict[str, Any] = {"piqa": _piqa_prompt}


def load_text_dataset(cfg: dict[str, Any]):
    """Load a Hugging Face dataset split as a list of text strings.

    Rows normally come from a single ``text_column``. Task datasets that spread
    a prompt over several columns instead name a builder via ``text_builder``
    (see :data:`TEXT_BUILDERS`), which is how PIQA-prompt calibration for H3 is
    fed through the same measurement pipeline as WikiText-2.
    """
    dataset_id = cfg["dataset_id"]
    dataset_config = cfg.get("dataset_config")

    if dataset_id == "wikitext" and dataset_config == "wikitext-2-raw-v1":
        ds = _load_wikitext2()
    else:
        ds = load_dataset(dataset_id, dataset_config)

    split = cfg.get("split", "test")
    column = cfg.get("text_column", "text")
    max_samples = cfg.get("max_samples")
    builder_name = cfg.get("text_builder")

    subset = ds[split]
    if max_samples is not None:
        subset = subset.select(range(min(int(max_samples), len(subset))))

    if builder_name:
        try:
            builder = TEXT_BUILDERS[builder_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown text_builder {builder_name!r}; known: {sorted(TEXT_BUILDERS)}"
            ) from exc
        texts = [builder(row) for row in subset]
        texts = [t for t in texts if t and t.strip()]
    else:
        texts = [row[column] for row in subset if row.get(column)]

    return texts, {
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "split": split,
        "text_builder": builder_name,
        "num_samples": len(texts),
    }


def build_calibration_blocks(
    texts: list[str],
    tokenizer: Any,
    *,
    seq_len: int = 512,
    max_blocks: int | None = 256,
    add_special_tokens: bool = False,
) -> list[torch.Tensor]:
    """Concatenate texts and split into uniform ``seq_len`` token blocks.

    WikiText rows are uneven (short headers, blank lines), so for activation
    calibration we join everything and chop into equal-length blocks. This gives
    even token coverage per forward pass, which keeps the per-neuron frequency
    estimate well-conditioned.

    Returns a list of ``[1, seq_len]`` ``input_ids`` tensors (the last partial
    block is dropped). ``max_blocks=None`` keeps all blocks.
    """
    joined = "\n\n".join(t for t in texts if t and t.strip())
    encoded = tokenizer(joined, add_special_tokens=add_special_tokens, return_tensors="pt")
    ids = encoded["input_ids"][0]
    num_blocks = ids.numel() // seq_len
    if max_blocks is not None:
        num_blocks = min(num_blocks, int(max_blocks))
    if num_blocks == 0:
        raise RuntimeError(
            f"Not enough calibration tokens ({ids.numel()}) for a single block of {seq_len}."
        )
    usable = ids[: num_blocks * seq_len].view(num_blocks, seq_len)
    return [usable[i : i + 1] for i in range(num_blocks)]
