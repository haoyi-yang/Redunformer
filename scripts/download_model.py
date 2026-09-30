#!/usr/bin/env python
"""Download a Hugging Face model repo (uses HF_TOKEN from environment)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, hf_hub_url, list_repo_files
from huggingface_hub.utils import build_hf_headers
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_LARGE_SUFFIXES = (".safetensors", ".bin", ".pt")


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _download_resumable(repo: str, filename: str, dest: Path, *, attempts: int = 10) -> Path:
    """Fetch one file, resuming from whatever is already on disk.

    A multi-gigabyte shard on a slow link gets truncated mid-stream often
    enough that a single dropped connection would otherwise throw away hours of
    progress, so each attempt re-issues a Range request from the current file
    size rather than starting over. HTTP 416 means the server has nothing left
    to send, i.e. the file is already complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(repo, filename)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        downloaded = dest.stat().st_size if dest.exists() else 0
        headers = build_hf_headers()
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"

        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                if response.status_code == 416 and downloaded > 0:
                    return dest
                response.raise_for_status()
                total = response.headers.get("content-length")
                total_size = int(total) + downloaded if total and total.isdigit() else None

                mode = "ab" if downloaded else "wb"
                with open(dest, mode) as handle, tqdm(
                    total=total_size,
                    initial=downloaded,
                    unit="B",
                    unit_scale=True,
                    desc=filename,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        bar.update(len(chunk))
            return dest
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            have = dest.stat().st_size if dest.exists() else 0
            if attempt == attempts:
                break
            backoff = min(30, 2**attempt)
            print(
                f"  attempt {attempt}/{attempts} dropped at {have:,} bytes "
                f"({type(exc).__name__}); resuming in {backoff}s",
                flush=True,
            )
            time.sleep(backoff)

    raise RuntimeError(
        f"Gave up on {filename} after {attempts} attempts; rerun to resume."
    ) from last_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HF model weights.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dotenv", default=str(REPO_ROOT / ".env"))
    parser.add_argument(
        "--local-dir",
        default="models/cache/Qwen3.5-4B",
        help="Directory for resumable HTTP downloads.",
    )
    parser.add_argument(
        "--method",
        choices=("resumable", "hub"),
        default="resumable",
        help="resumable: HTTP range downloads; hub: huggingface_hub cache",
    )
    args = parser.parse_args()

    _load_dotenv(Path(args.dotenv))
    token_set = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))
    print(f"HF token set: {token_set}", flush=True)

    repo = args.model_id
    # Nested directories in a model repo hold alternative formats (onnx/,
    # openvino/, coreml/) that duplicate the weights in another encoding. We
    # only want the PyTorch checkpoint, so top-level files only - otherwise the
    # download silently pulls the model twice.
    files = sorted(
        f
        for f in list_repo_files(repo)
        if "/" not in f
        and (f.endswith(".safetensors") or f.endswith(".json") or f.endswith(".txt"))
    )
    print(f"Downloading {len(files)} files from {repo} ...", flush=True)

    out = Path(args.local_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, filename in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {filename}", flush=True)
        if args.method == "hub" and not filename.endswith(_LARGE_SUFFIXES):
            path = hf_hub_download(repo, filename)
            print(f"  -> {path}", flush=True)
            continue

        dest = out / filename
        if filename.endswith(_LARGE_SUFFIXES):
            path = _download_resumable(repo, filename, dest)
        else:
            if dest.exists() and dest.stat().st_size > 0:
                path = dest
            else:
                path = Path(hf_hub_download(repo, filename))
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(path.read_bytes())
        print(f"  -> {path}", flush=True)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
