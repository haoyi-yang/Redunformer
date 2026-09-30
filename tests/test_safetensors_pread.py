"""Tests for the non-mmap safetensors reader.

The reader exists because ``safe_open``'s mmap crashes the interpreter on
Windows under memory pressure, so what matters is that it returns byte-identical
tensors through the same API transformers calls.
"""

from __future__ import annotations

import pytest
import torch
from redundancy.safetensors_pread import PreadSafeFile, enable_pread_safetensors
from safetensors import safe_open
from safetensors.torch import save_file


@pytest.fixture
def checkpoint(tmp_path):
    tensors = {
        "embed.weight": torch.randn(37, 8, dtype=torch.float32),
        "layer.0.weight": torch.randn(16, 8).to(torch.float16),
        "layer.0.bias": torch.randn(16).to(torch.bfloat16),
        "counts": torch.arange(12, dtype=torch.int64).reshape(3, 4),
        "flag": torch.tensor([True, False, True]),
        "scalar": torch.tensor(2.5),
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, path, metadata={"format": "pt"})
    return path, tensors


def test_keys_match_the_reference_reader(checkpoint):
    path, tensors = checkpoint
    with PreadSafeFile(path) as f:
        assert sorted(f.keys()) == sorted(tensors)


def test_tensors_are_identical_to_the_reference_reader(checkpoint):
    path, tensors = checkpoint
    with PreadSafeFile(path) as ours, safe_open(path, framework="pt") as theirs:
        for name in tensors:
            mine = ours.get_tensor(name)
            reference = theirs.get_tensor(name)
            assert mine.dtype == reference.dtype
            assert mine.shape == reference.shape
            assert torch.equal(mine, reference), name


def test_slice_reports_shape_and_dtype_without_reading(checkpoint):
    path, _ = checkpoint
    with PreadSafeFile(path) as ours, safe_open(path, framework="pt") as theirs:
        mine, reference = ours.get_slice("layer.0.weight"), theirs.get_slice("layer.0.weight")
        assert mine.get_shape() == reference.get_shape()
        assert mine.get_dtype() == reference.get_dtype()


def test_ellipsis_materializes_the_whole_tensor(checkpoint):
    """This is exactly how transformers loads each weight."""
    path, tensors = checkpoint
    with PreadSafeFile(path) as f:
        assert torch.equal(f.get_slice("embed.weight")[...], tensors["embed.weight"])


def test_partial_slicing_matches_the_full_tensor(checkpoint):
    path, tensors = checkpoint
    with PreadSafeFile(path) as f:
        assert torch.equal(f.get_slice("counts")[1:], tensors["counts"][1:])


def test_materialized_tensors_are_writable(checkpoint):
    """torch.frombuffer over a read-only buffer warns and yields a fragile tensor."""
    path, _ = checkpoint
    with PreadSafeFile(path) as f:
        tensor = f.get_tensor("embed.weight")
    tensor[0, 0] = 1.5
    assert tensor[0, 0] == 1.5


def test_metadata_is_preserved(checkpoint):
    path, _ = checkpoint
    with PreadSafeFile(path) as f:
        assert f.metadata() == {"format": "pt"}


def test_tensors_outlive_the_file_handle(checkpoint):
    """Transformers closes shards only after loading, but never re-reads a
    materialized tensor - a stale mmap is what makes the stock reader unsafe."""
    path, tensors = checkpoint
    with PreadSafeFile(path) as f:
        tensor = f.get_tensor("embed.weight")
    assert torch.equal(tensor, tensors["embed.weight"])


def test_unsupported_framework_is_rejected(tmp_path):
    save_file({"a": torch.zeros(2)}, tmp_path / "m.safetensors")
    with pytest.raises(ValueError, match="framework"):
        PreadSafeFile(tmp_path / "m.safetensors", framework="np")


def test_truncated_file_raises_instead_of_crashing(tmp_path, checkpoint):
    """The whole point of the reader: a bad read must be catchable."""
    path, _ = checkpoint
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    with PreadSafeFile(truncated) as f:
        with pytest.raises((EOFError, ValueError)):
            for name in f.keys():
                f.get_tensor(name)


def test_enable_is_idempotent_and_patches_transformers():
    from transformers import modeling_utils

    original = modeling_utils.safe_open
    try:
        assert enable_pread_safetensors(force=True)
        patched = modeling_utils.safe_open
        assert patched is PreadSafeFile
        assert enable_pread_safetensors(force=True)
        assert modeling_utils.safe_open is patched
    finally:
        modeling_utils.safe_open = original


def test_opt_out_env_var_is_respected(monkeypatch):
    from transformers import modeling_utils

    monkeypatch.setenv("REDUNDANCY_DISABLE_PREAD_SAFETENSORS", "1")
    original = modeling_utils.safe_open
    assert not enable_pread_safetensors(force=True)
    assert modeling_utils.safe_open is original
