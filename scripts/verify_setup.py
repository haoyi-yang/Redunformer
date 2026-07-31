#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "models"


def check_import(name: str) -> str:
    module = importlib.import_module(name)
    return getattr(module, "__version__", "ok")


def check_configs() -> list[str]:
    loaded = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        assert cfg["pretrained"], f"missing pretrained in {path.name}"
        assert cfg["tasks"], f"missing tasks in {path.name}"
        loaded.append(path.name)
    return loaded


def main() -> int:
    print("Redunformer setup verification")
    print(f"project root: {ROOT}")

    packages = [
        "torch",
        "transformers",
        "datasets",
        "lm_eval",
        "numpy",
        "matplotlib",
    ]
    for package in packages:
        version = check_import(package)
        print(f"  ok  {package} ({version})")

    configs = check_configs()
    print(f"  ok  configs ({', '.join(configs)})")

    sys.path.insert(0, str(ROOT / "src"))
    from redundancy.pruning import sparsegpt as sparsegpt_mod

    assert hasattr(sparsegpt_mod, "SparseGPT")
    assert hasattr(sparsegpt_mod, "sparsegpt_prune_model")
    assert sparsegpt_mod.ALGORITHM_NAME == "sparsegpt"
    print("  ok  sparsegpt pruning module")

    print("Verification passed. No models were downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
