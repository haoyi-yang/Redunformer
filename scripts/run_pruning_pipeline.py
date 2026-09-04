from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from redundancy.models import load_model_and_tokenizer


MODELS = {
    "1": ("Qwen/Qwen3-4B", "qwen3-4b"),
    "2": ("Qwen/Qwen3-1.7B", "qwen3-1.7b"),
    "3": ("gpt2", "gpt2"),
}

# HF id or short name -> (hf_id, short_name)
MODEL_ALIASES = {
    "gpt2": ("gpt2", "gpt2"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "qwen3-1.7b"),
    "qwen3-4b": ("Qwen/Qwen3-4B", "qwen3-4b"),
    "Qwen/Qwen3-1.7B": ("Qwen/Qwen3-1.7B", "qwen3-1.7b"),
    "Qwen/Qwen3-4B": ("Qwen/Qwen3-4B", "qwen3-4b"),
}


def discover_pruning_algorithms():
    """Top-level pruning modules plus DSnoT-family under pruning/dsnot/."""
    pruning_dir = Path("src/redundancy/pruning")
    algorithms: list[str] = []
    for file in sorted(pruning_dir.glob("*.py")):
        if file.name.startswith("__"):
            continue
        algorithms.append(file.stem)
    dsnot_dir = pruning_dir / "dsnot"
    if dsnot_dir.is_dir():
        for file in sorted(dsnot_dir.glob("*.py")):
            if file.name.startswith("__"):
                continue
            algorithms.append(file.stem)
    return algorithms


def load_algorithm_module(algo_name: str):
    """Load a top-level or ``pruning/dsnot/`` algorithm module by stem name."""
    pruning_dir = Path("src/redundancy/pruning")
    dsnot_file = pruning_dir / "dsnot" / f"{algo_name}.py"
    top_file = pruning_dir / f"{algo_name}.py"
    # Prefer the real .py module path. Important: ``dsnot`` is both a package
    # directory and ``dsnot/dsnot.py`` — importing ``redundancy.pruning.dsnot``
    # would load the package, not the refine algorithm.
    if dsnot_file.is_file():
        return importlib.import_module(f"redundancy.pruning.dsnot.{algo_name}")
    if top_file.is_file():
        return importlib.import_module(f"redundancy.pruning.{algo_name}")
    raise ModuleNotFoundError(
        f"No pruning algorithm module named {algo_name!r} "
        f"(looked in {top_file} and {dsnot_file})"
    )


def resolve_prune_fn(module):
    if hasattr(module, "prune") and callable(module.prune):
        return module.prune
    for name in dir(module):
        if name.endswith("_prune_model"):
            return getattr(module, name)
    raise RuntimeError(f"No prune() / *_prune_model found in {module.__name__}")


def resolve_model(model_arg: str) -> tuple[str, str]:
    if model_arg in MODEL_ALIASES:
        return MODEL_ALIASES[model_arg]
    if model_arg in MODELS:
        return MODELS[model_arg]
    # Arbitrary HF id / path
    short = model_arg.split("/")[-1].lower()
    return model_arg, short


def list_pruned_checkpoints() -> list[Path]:
    root = Path("experiments/pruned")
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    )


def infer_dense_from_name(name: str) -> str:
    lower = name.lower()
    if "qwen3-4b" in lower:
        return "Qwen/Qwen3-4B"
    if "qwen3-1.7b" in lower or "qwen3-1.7" in lower:
        return "Qwen/Qwen3-1.7B"
    if "gpt2" in lower:
        return "gpt2"
    return ""


def is_local_checkpoint(model_id: str) -> bool:
    path = Path(model_id)
    return path.is_dir() and (path / "config.json").exists()


def choose_model():
    print("\nAvailable models:\n")
    print("1) Qwen3-4B")
    print("2) Qwen3-1.7B")
    print("3) GPT2")
    ckpts = list_pruned_checkpoints()
    for idx, path in enumerate(ckpts, start=4):
        print(f"{idx}) {path}  [pruned]")
    choice = input("\nSelect model: ").strip()
    if choice in MODELS:
        return MODELS[choice]
    try:
        n = int(choice)
    except ValueError as exc:
        raise ValueError("Invalid model selection") from exc
    if n >= 4 and n - 4 < len(ckpts):
        path = ckpts[n - 4]
        return str(path), path.name
    raise ValueError("Invalid model selection")


def choose_algorithm():
    algorithms = discover_pruning_algorithms()
    print("\nAvailable pruning algorithms:\n")
    for idx, algo in enumerate(algorithms, start=1):
        print(f"{idx}) {algo}")
    choice = int(input("\nSelect algorithm: "))
    return algorithms[choice - 1]


def prompt_parameters(module) -> dict:
    params: dict = {}
    parameters = getattr(module, "PARAMETERS", None)
    if parameters:
        for name, meta in parameters.items():
            prompt = meta.get("prompt", name)
            default = meta.get("default")
            typ = meta.get("type", str)
            suffix = f" [{default}]" if default is not None else ""
            raw = input(f"{prompt}{suffix}: ").strip()
            if raw == "" and default is not None:
                params[name] = default
            else:
                params[name] = typ(raw)
        return params
    sparsity = float(input("\nSparsity (0-1): ").strip())
    return {"sparsity": sparsity}


def parameters_from_args(module, args: argparse.Namespace) -> dict:
    """Fill PARAMETERS from CLI flags; fall back to module defaults (no prompts)."""
    parameters = getattr(module, "PARAMETERS", None)
    if not parameters:
        sparsity = args.sparsity if args.sparsity is not None else 0.5
        return {"sparsity": float(sparsity)}

    params: dict = {}
    for name, meta in parameters.items():
        typ = meta.get("type", str)
        default = meta.get("default")
        cli_val = getattr(args, name, None)
        if cli_val is not None:
            params[name] = typ(cli_val)
        elif default is not None:
            params[name] = default
        else:
            raise ValueError(
                f"Missing required parameter --{name} (no default in {module.__name__})"
            )
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pruning pipeline. With --model and --algorithm runs non-interactively; "
            "otherwise prompts for missing values."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF model id / alias (gpt2, qwen3-1.7b, qwen3-4b, ...)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Pruning module name under src/redundancy/pruning/",
    )
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--nsamples", type=int, default=None)
    parser.add_argument("--seqlen", type=int, default=None)
    parser.add_argument("--blocksize", type=int, default=None)
    parser.add_argument("--percdamp", type=float, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prunen", type=int, default=None)
    parser.add_argument("--prunem", type=int, default=None)
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="DSnoT initial metric (wanda|magnitude|sparsegpt)",
    )
    parser.add_argument("--cycles", type=int, default=None, help="DSnoT max cycles T")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="DSnoT reconstruction-error stop threshold",
    )
    parser.add_argument(
        "--var_power",
        type=float,
        default=None,
        help="DSnoT power of activation variance in the grow score",
    )
    parser.add_argument(
        "--same_sign",
        type=int,
        default=None,
        help="DSnoT: reject swaps that flip error sign (1/0)",
    )
    parser.add_argument(
        "--skip_layer",
        type=str,
        default=None,
        help="DSnoT: skip refinement on this module prefix (none|mlp|self_attn)",
    )
    parser.add_argument(
        "--dense",
        type=str,
        default=None,
        help="Dense original HF id/path when refining a saved sparse checkpoint",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only prune + save; do not run lm-eval",
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    interactive = args.model is None or args.algorithm is None

    if args.model is None:
        model_id, model_name = choose_model()
    else:
        model_id, model_name = resolve_model(args.model)

    if args.algorithm is None:
        algorithm = choose_algorithm()
    else:
        algorithm = args.algorithm
        available = discover_pruning_algorithms()
        if algorithm not in available:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Available: {', '.join(available)}"
            )

    module = load_algorithm_module(algorithm)
    algo_label = getattr(module, "ALGORITHM_NAME", algorithm)

    if interactive:
        params = prompt_parameters(module)
    else:
        params = parameters_from_args(module, args)
        print(
            f"Non-interactive: model={model_id}, algorithm={algorithm}, params={params}"
        )

    if algorithm == "dsnot":
        dense = str(params.get("dense") or "").strip()
        if not dense and is_local_checkpoint(model_id):
            dense = infer_dense_from_name(model_name)
            if dense:
                print(
                    f"Sparse checkpoint {model_id} → DSnoT refine only "
                    f"(dense original {dense})"
                )
                params["dense"] = dense
                params["base"] = "existing"
        elif dense:
            params["base"] = "existing"

    needs_tokenizer = algorithm in {
        "sparsegpt",
        "sparse_dsnot",
        "wanda",
        "dsnot",
    }

    print("\nLoading model...\n")
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        dtype=(
            "float16"
            if algorithm in {"sparsegpt", "sparse_dsnot"}
            else "float32"
        ),
        device_map="cpu",
    )

    prune_fn = resolve_prune_fn(module)
    print(f"\nRunning {algo_label} pruning...\n")

    call_kwargs = dict(params)
    if needs_tokenizer:
        call_kwargs["tokenizer"] = tokenizer

    prune_fn(model=model, **call_kwargs)

    sparsity = float(params.get("sparsity", 0.0))
    percent = int(sparsity * 100)
    if algorithm == "dsnot" and str(params.get("base", "")).lower() == "existing":
        output_name = f"{model_name}-dsnot"
    else:
        output_name = f"{model_name}-{algorithm}{percent}"
    output_dir = f"experiments/pruned/{output_name}"

    print(f"\nSaving model to {output_dir}\n")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("\nModel saved.")

    config = {
        "name": output_name,
        "model": "hf",
        "pretrained": output_dir,
        "dtype": "bfloat16",
        "device_map": "auto",
        "tasks": ["hellaswag", "piqa", "arc_easy"],
        "batch_size": "auto",
        "seed": 42,
        "output_dir": "experiments/baseline",
    }

    config_path = f"configs/models/{output_name}.yaml"
    Path("configs/models").mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    print(f"\nCreated config: {config_path}")

    if args.skip_eval:
        print("\nSkipping evaluation (--skip-eval).")
        print("\nPipeline finished.")
        return

    print("\nStarting evaluation...\n")
    subprocess.run(
        [sys.executable, "scripts/run_baseline.py", "--config", config_path],
        check=True,
    )
    print("\nPipeline finished.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
