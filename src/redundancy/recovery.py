"""Short LoRA fine-tuning after masking (Weeks 11–12 recovery).

Weeks 9–10 showed that importance-guided masking is far cheaper than random, but
still costs perplexity at every ratio. The recovery question is whether that
remaining cost is *real capacity loss* or just a network knocked off its
operating point: if a few hundred LoRA steps restore the original perplexity,
the masked neurons were genuinely redundant and their role was already covered
elsewhere (proposal Q1, item 4).

Two design choices worth stating:

* **Masks stay active during training.** The masker's pre-hooks fire in training
  exactly as in evaluation, so the adapters learn to route around the removed
  channels rather than through them.
* **LoRA does not target ``down_proj``.** A masked channel feeds ``down_proj``
  with zeros, so its columns receive no gradient and could not be repaired
  anyway; leaving ``down_proj`` unwrapped also keeps the mask hooks attached to
  plain ``Linear`` modules. Recovery therefore happens in the projections that
  *feed* the surviving neurons.

Any comparison needs the **unmasked control**: the same number of steps on the
unmasked model. Without it, a perplexity gain cannot be separated from ordinary
in-domain adaptation to the fine-tuning slice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

# LoRA on the projections feeding the FFN and on attention; ``down_proj`` is
# deliberately absent (see module docstring).
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
)


@dataclass
class RecoveryConfig:
    """Hyper-parameters for a short LoRA recovery run."""

    rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES
    learning_rate: float = 2e-4
    steps: int = 200
    batch_size: int = 1
    grad_accum: int = 8
    warmup_steps: int = 20
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    seed: int = 42
    gradient_checkpointing: bool = False

    def summary(self) -> dict:
        return {
            "rank": self.rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "gradient_checkpointing": self.gradient_checkpointing,
        }


@dataclass
class TrainingHistory:
    """Loss trace of a recovery run."""

    losses: list[float] = field(default_factory=list)
    steps: int = 0

    @property
    def first_loss(self) -> float | None:
        return self.losses[0] if self.losses else None

    @property
    def final_loss(self) -> float | None:
        return self.losses[-1] if self.losses else None

    def summary(self) -> dict:
        out: dict[str, Any] = {"steps": self.steps, "losses": self.losses}
        if self.losses:
            tail = self.losses[-max(1, len(self.losses) // 5) :]
            out["first_loss"] = self.first_loss
            out["final_loss"] = self.final_loss
            out["mean_tail_loss"] = float(sum(tail) / len(tail))
        return out


def resolve_target_modules(model: torch.nn.Module, requested: Sequence[str]) -> list[str]:
    """Keep only requested LoRA targets that this architecture actually has."""
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    return [target for target in requested if target in names]


def build_lora_model(model: torch.nn.Module, cfg: RecoveryConfig, *, is_4bit: bool = False):
    """Wrap ``model`` with LoRA adapters, freezing all base weights."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    targets = resolve_target_modules(model, cfg.target_modules)
    if not targets:
        raise RuntimeError(
            f"None of the requested LoRA targets exist in this model: {list(cfg.target_modules)}"
        )
    if is_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    lora_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    if cfg.gradient_checkpointing and hasattr(peft_model, "gradient_checkpointing_enable"):
        peft_model.gradient_checkpointing_enable()
        peft_model.enable_input_require_grads()
    return peft_model


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    """``(trainable, total)`` parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _lr_scale(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay, as a multiplier on the base rate."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    remaining = max(total - warmup, 1)
    progress = min(max(step - warmup, 0) / remaining, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_lora(
    model: torch.nn.Module,
    blocks: Sequence[torch.Tensor],
    cfg: RecoveryConfig,
    *,
    device: torch.device | str = "cpu",
    progress: Callable[[int, float], None] | None = None,
) -> TrainingHistory:
    """Run ``cfg.steps`` optimizer steps of causal-LM LoRA training on ``blocks``.

    ``blocks`` are ``[1, seq_len]`` token tensors (the same calibration blocks
    the measurement stage uses). Blocks are consumed in a shuffled, repeating
    order so a short run still sees varied text.
    """
    if not blocks:
        raise ValueError("No training blocks provided.")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters; was the model wrapped with LoRA?")

    generator = torch.Generator().manual_seed(cfg.seed)
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    history = TrainingHistory()

    order: list[int] = []

    def next_batch_ids() -> list[int]:
        ids: list[int] = []
        while len(ids) < cfg.batch_size:
            if not order:
                order.extend(torch.randperm(len(blocks), generator=generator).tolist())
            ids.append(order.pop())
        return ids

    model.train()
    for step in range(cfg.steps):
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate * _lr_scale(step, cfg.steps, cfg.warmup_steps)

        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(cfg.grad_accum):
            batch = torch.cat([blocks[i] for i in next_batch_ids()], dim=0).to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss / cfg.grad_accum
            loss.backward()
            accumulated += float(outputs.loss.detach())

        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        optimizer.step()

        mean_loss = accumulated / cfg.grad_accum
        history.losses.append(mean_loss)
        history.steps = step + 1
        if progress is not None:
            progress(step + 1, mean_loss)

    model.eval()
    return history


def recovery_fraction(baseline: float, masked: float, recovered: float) -> float | None:
    """Share of the masking damage that recovery undoes.

    ``1.0`` means the original perplexity is fully restored, ``0.0`` that
    fine-tuning changed nothing, and negative that it made things worse.
    Returns ``None`` when masking did not cost anything to begin with.
    """
    damage = masked - baseline
    if abs(damage) < 1e-9:
        return None
    return (masked - recovered) / damage
