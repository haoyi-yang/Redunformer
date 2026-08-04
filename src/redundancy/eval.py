from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_model_args(cfg: dict[str, Any]) -> str:
    parts = [f"pretrained={cfg['pretrained']}"]
    if cfg.get("dtype"):
        parts.append(f"dtype={cfg['dtype']}")
    if cfg.get("device_map"):
        parts.append(f"device_map={cfg['device_map']}")
    return ",".join(parts)


def build_eval_command(cfg: dict[str, Any], *, limit: float | None = None) -> list[str]:
    effective_limit = limit if limit is not None else cfg.get("limit")
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        cfg.get("model", "hf"),
        "--model_args",
        build_model_args(cfg),
        "--tasks",
        ",".join(cfg["tasks"]),
        "--batch_size",
        str(cfg.get("batch_size", "auto")),
    ]
    if cfg.get("seed") is not None:
        cmd.extend(["--seed", str(cfg["seed"])])
    if effective_limit is not None:
        cmd.extend(["--limit", str(effective_limit)])
    return cmd


def run_baseline_eval(
    cfg: dict[str, Any],
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    limit: float | None = None,
) -> Path:
    output_root = Path(output_dir or cfg.get("output_dir", "experiments/baseline"))
    output_root.mkdir(parents=True, exist_ok=True)

    model_slug = cfg["pretrained"].replace("/", "_")
    output_path = output_root / f"{model_slug}.json"

    cmd = build_eval_command(cfg, limit=limit)
    cmd.extend(["--output_path", str(output_path)])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    meta = {
        "name": cfg.get("name"),
        "config_path": str(config_path),
        "command": cmd,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": cfg.get("model", "hf"),
        "pretrained": cfg["pretrained"],
        "tasks": cfg["tasks"],
        "seed": cfg.get("seed"),
        "limit": limit if limit is not None else cfg.get("limit"),
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output_path
