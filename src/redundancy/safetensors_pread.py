"""A non-mmap safetensors reader, used to load large checkpoints on Windows.

``safetensors.safe_open`` maps the checkpoint with
``torch.UntypedStorage.from_file``. Reading a tensor is then an ordinary memory
access, and if Windows cannot service the resulting page fault the process dies
with an access violation (0xC0000005) instead of raising - the traceback ends
in ``UntypedStorage.__getitem__`` with no Python-level error. It reproduces
here when loading a sharded 7B checkpoint while free RAM is well below the
mapped size, and it takes the whole interpreter down, so it cannot be caught
and retried.

Transformers exposes ``disable_mmap``, but that reads each shard fully into RAM
before parsing, which needs more memory than the machines hitting the bug have.
Instead this module reads each tensor's byte range on demand with ordinary file
reads, keeping the one-tensor-at-a-time behaviour that lazy loading and
on-the-fly quantization rely on.

``PreadSafeFile`` is a drop-in for the parts of ``safe_open`` that transformers
uses, so :func:`enable_pread_safetensors` can swap it in at the call site.
"""

from __future__ import annotations

import json
import os
import struct
import threading
from pathlib import Path
from typing import Any

import torch

_HEADER_SIZE_BYTES = 8
_METADATA_KEY = "__metadata__"

# safetensors spells dtypes with its own short codes.
_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    """Return the parsed header and the offset where tensor data starts."""
    with open(path, "rb") as handle:
        raw_size = handle.read(_HEADER_SIZE_BYTES)
        if len(raw_size) != _HEADER_SIZE_BYTES:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (header_size,) = struct.unpack("<Q", raw_size)
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"{path} has a truncated header")
    return json.loads(header_bytes), _HEADER_SIZE_BYTES + header_size


class PreadSlice:
    """Lazy handle on one tensor, materialized only when indexed."""

    def __init__(self, parent: PreadSafeFile, name: str, info: dict[str, Any], data_start: int):
        self._parent = parent
        self._name = name
        self._dtype_str = info["dtype"]
        self._shape = list(info["shape"])
        begin, end = info["data_offsets"]
        self._offset = data_start + begin
        self._nbytes = end - begin

    def get_shape(self) -> list[int]:
        return list(self._shape)

    def get_dtype(self) -> str:
        return self._dtype_str

    def _materialize(self) -> torch.Tensor:
        dtype = _DTYPES.get(self._dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported safetensors dtype {self._dtype_str!r} for {self._name}")
        if self._nbytes == 0:
            return torch.empty(self._shape, dtype=dtype)
        # A bytearray is writable, so frombuffer hands back a normal tensor
        # instead of warning about a read-only buffer, and the tensor keeps the
        # buffer alive for us.
        buffer = bytearray(self._nbytes)
        self._parent._read_into(self._offset, buffer)
        return torch.frombuffer(buffer, dtype=dtype).reshape(self._shape)

    def __getitem__(self, key: Any) -> torch.Tensor:
        tensor = self._materialize()
        # Transformers materializes with `slice[...]`; anything narrower is
        # only used by tensor-parallel loading, where correctness matters more
        # than avoiding the full read.
        if key is Ellipsis:
            return tensor
        return tensor[key]


class PreadSafeFile:
    """Drop-in replacement for ``safetensors.safe_open`` that avoids mmap."""

    def __init__(self, filename: str | os.PathLike[str], framework: str = "pt", device: str = "cpu"):
        if framework != "pt":
            raise ValueError(f"Only the 'pt' framework is supported, got {framework!r}")
        if device not in ("cpu", None):
            raise ValueError(f"Only CPU reads are supported, got device={device!r}")
        self._path = Path(filename)
        header, self._data_start = _read_header(self._path)
        self._metadata = header.get(_METADATA_KEY)
        self._header = {k: v for k, v in header.items() if k != _METADATA_KEY}
        # One handle shared by every read, so loading a shard does not reopen
        # the file hundreds of times. Seek and read must not interleave.
        self._handle = open(self._path, "rb", buffering=0)
        self._lock = threading.Lock()

    def _read_into(self, offset: int, buffer: bytearray) -> None:
        view = memoryview(buffer)
        with self._lock:
            self._handle.seek(offset)
            read = 0
            while read < len(buffer):
                got = self._handle.readinto(view[read:])
                if not got:
                    raise EOFError(
                        f"{self._path} ended after {offset + read} bytes; expected "
                        f"{len(buffer) - read} more"
                    )
                read += got

    def keys(self) -> list[str]:
        return list(self._header.keys())

    def metadata(self) -> dict[str, str] | None:
        return self._metadata

    def get_slice(self, name: str) -> PreadSlice:
        return PreadSlice(self, name, self._header[name], self._data_start)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.get_slice(name)[...]

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> PreadSafeFile:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def enable_pread_safetensors(force: bool = False) -> bool:
    """Route transformers' checkpoint reads through :class:`PreadSafeFile`.

    Only applied on Windows, where the mmap crash occurs, unless ``force`` is
    set. Set ``REDUNDANCY_DISABLE_PREAD_SAFETENSORS=1`` to keep the stock mmap
    reader. Returns whether the replacement is in effect.
    """
    if os.environ.get("REDUNDANCY_DISABLE_PREAD_SAFETENSORS") == "1":
        return False
    if not force and os.name != "nt":
        return False

    from transformers import modeling_utils

    if getattr(modeling_utils.safe_open, "_is_pread_reader", False):
        return True
    PreadSafeFile._is_pread_reader = True
    modeling_utils.safe_open = PreadSafeFile
    return True
