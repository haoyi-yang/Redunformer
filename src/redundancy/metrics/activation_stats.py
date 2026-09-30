"""Streaming per-neuron activation statistics for FFN redundancy measurement.

For each FFN layer we maintain *exact* streaming aggregates over the full
calibration set (token count, mean ``|activation|``, RMS, and a fixed-threshold
firing frequency) plus a small **reservoir** subsample of activation rows. The
reservoir is used for the quantities that cannot be counted exactly in a single
streaming pass without first knowing the layer scale:

* the **RMS-relative** firing frequency ``f_i = P(|a_i| > alpha * RMS_layer)`` at
  several ``alpha`` values (the "lazy neuron" probe, Li et al. 2022), and
* pairwise neuron **correlation** (computed separately, see ``correlation.py``).

Because post-SwiGLU activations are rarely near zero, firing frequency saturates
near 1.0 for these models. The more discriminative redundancy signal is the
**activation-aware importance** ``rms_i * ||W_down[:, i]||`` (LLM-Pruner / Wanda
style): the expected contribution of neuron ``i`` to the residual stream. The
weight column norm is supplied after the run via
:meth:`LayerActivationResult.set_weight_col_norm`.

Memory is bounded by ``max_reservoir_tokens * width`` per layer (stored in
float16), not by the calibration-set size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

_DEFAULT_ALPHAS = (1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1)


@dataclass
class LayerActivationResult:
    """Finalized per-neuron statistics for one FFN layer."""

    name: str
    layer_index: int
    width: int
    num_tokens: int
    layer_rms: float
    fixed_eps: float
    rms_alpha: float
    rms_eps: float
    mean_abs: np.ndarray
    rms: np.ndarray
    freq_fixed: np.ndarray
    freq_rms: np.ndarray
    reservoir_tokens: int
    rms_alphas: list[float] = field(default_factory=list)
    freq_by_alpha: dict[float, np.ndarray] = field(default_factory=dict)
    # Activation-aware importance (filled after the run once weights are known).
    col_norm: np.ndarray | None = None
    importance: np.ndarray | None = None

    def lazy_fraction(self, freq_threshold: float = 0.0) -> float:
        """Fraction of neurons whose primary RMS-relative frequency <= threshold."""
        return float(np.mean(self.freq_rms <= freq_threshold))

    def lazy_fraction_at_alpha(self, alpha: float, freq_threshold: float = 0.01) -> float:
        """Fraction of neurons that fire on <= ``freq_threshold`` of tokens at ``alpha``."""
        freq = self.freq_by_alpha.get(alpha)
        if freq is None:
            return self.lazy_fraction(freq_threshold)
        return float(np.mean(freq <= freq_threshold))

    def set_weight_col_norm(self, col_norm: np.ndarray | None) -> None:
        """Attach the down-proj input-column norms and derive neuron importance.

        ``importance_i = RMS(h_i) * ||W_down[:, i]||`` approximates the expected
        L2 contribution of neuron ``i`` to the FFN output. When weights are
        unavailable (e.g. an un-dequantizable quantized layer) we fall back to
        activation RMS alone.
        """
        if col_norm is not None and col_norm.shape == self.rms.shape:
            self.col_norm = col_norm.astype(np.float64)
            self.importance = self.rms * self.col_norm
        else:
            self.col_norm = None
            self.importance = self.rms.copy()

    @property
    def importance_vector(self) -> np.ndarray:
        return self.importance if self.importance is not None else self.rms

    def importance_share_bottom(self, frac: float = 0.1) -> float:
        """Share of total neuron importance held by the bottom ``frac`` neurons.

        A small value means the least-important ``frac`` of neurons carry little
        of the layer's output contribution, i.e. they are cheap to remove. This
        is the non-saturated removability proxy that replaces firing frequency.
        """
        imp = self.importance_vector
        total = float(imp.sum())
        if total <= 0 or imp.size == 0:
            return 0.0
        k = max(1, int(round(frac * imp.size)))
        bottom = np.sort(imp)[:k]
        return float(bottom.sum() / total)

    def summary(self) -> dict:
        out = {
            "name": self.name,
            "layer_index": self.layer_index,
            "width": self.width,
            "num_tokens": self.num_tokens,
            "layer_rms": self.layer_rms,
            "fixed_eps": self.fixed_eps,
            "rms_alpha": self.rms_alpha,
            "rms_eps": self.rms_eps,
            "reservoir_tokens": self.reservoir_tokens,
            "mean_freq_rms": float(self.freq_rms.mean()),
            "mean_freq_fixed": float(self.freq_fixed.mean()),
            "mean_abs": float(self.mean_abs.mean()),
            "lazy_fraction_le_1pct": self.lazy_fraction(0.01),
            "lazy_fraction_zero": self.lazy_fraction(0.0),
            "lazy_fraction_by_alpha_le_1pct": {
                str(a): self.lazy_fraction_at_alpha(a, 0.01) for a in self.rms_alphas
            },
            "col_norm_available": self.col_norm is not None,
        }
        imp = self.importance_vector
        out.update(
            {
                "mean_importance": float(imp.mean()),
                "median_importance": float(np.median(imp)),
                "importance_share_bottom_10pct": self.importance_share_bottom(0.10),
                "importance_share_bottom_25pct": self.importance_share_bottom(0.25),
            }
        )
        return out


@dataclass
class _LayerAccumulator:
    width: int
    fixed_eps: float
    max_reservoir_tokens: int
    rng: np.random.Generator
    num_tokens: int = 0
    sum_abs: torch.Tensor = field(default=None)  # type: ignore[assignment]
    sum_sq: torch.Tensor = field(default=None)  # type: ignore[assignment]
    count_fixed: torch.Tensor = field(default=None)  # type: ignore[assignment]
    reservoir: torch.Tensor = field(default=None)  # type: ignore[assignment]
    _filled: int = 0
    _seen: int = 0

    def __post_init__(self) -> None:
        self.sum_abs = torch.zeros(self.width, dtype=torch.float64)
        self.sum_sq = torch.zeros(self.width, dtype=torch.float64)
        self.count_fixed = torch.zeros(self.width, dtype=torch.float64)
        if self.max_reservoir_tokens > 0:
            self.reservoir = torch.zeros(
                (self.max_reservoir_tokens, self.width), dtype=torch.float16
            )

    def update(self, act: torch.Tensor) -> None:
        # act: [tokens, width] float32 (CPU).
        act64 = act.to(torch.float64)
        self.sum_abs += act64.abs().sum(dim=0)
        self.sum_sq += act64.pow(2).sum(dim=0)
        self.count_fixed += (act.abs() > self.fixed_eps).sum(dim=0, dtype=torch.float64)
        self._reservoir_update(act)
        self.num_tokens += act.shape[0]

    def _reservoir_update(self, act: torch.Tensor) -> None:
        """Vectorized Algorithm-R reservoir sampling over token rows.

        Deterministic via ``self.rng``. When several rows in a batch map to the
        same reservoir slot the last write wins; this is a negligible bias for a
        calibration subsample and keeps the update fully vectorized.
        """
        if self.reservoir is None:
            return
        cap = self.max_reservoir_tokens
        half = act.to(torch.float16)
        m = half.shape[0]

        start = 0
        if self._filled < cap:
            take = min(cap - self._filled, m)
            self.reservoir[self._filled : self._filled + take] = half[:take]
            self._filled += take
            self._seen += take
            start = take

        if start < m:
            remaining = half[start:]
            r = remaining.shape[0]
            # Global 1-indexed count after including each remaining row.
            seen_after = self._seen + np.arange(1, r + 1)
            j = np.floor(self.rng.random(r) * seen_after).astype(np.int64)
            accept = j < cap
            if accept.any():
                rows = np.nonzero(accept)[0]
                slots = torch.from_numpy(j[accept])
                self.reservoir[slots] = remaining[torch.from_numpy(rows)]
            self._seen += r

    def finalize(
        self,
        name: str,
        layer_index: int,
        rms_alpha: float,
        rms_alphas: Sequence[float],
    ) -> LayerActivationResult:
        n = max(self.num_tokens, 1)
        mean_abs = (self.sum_abs / n).numpy()
        per_neuron_ms = (self.sum_sq / n).numpy()
        rms = np.sqrt(per_neuron_ms)
        freq_fixed = (self.count_fixed / n).numpy()
        layer_rms = float(np.sqrt(per_neuron_ms.mean())) if self.width else 0.0

        reservoir = self.reservoir_matrix()
        freq_by_alpha: dict[float, np.ndarray] = {}
        if reservoir is not None and reservoir.shape[0] > 0:
            abs_res = np.abs(reservoir)
            for alpha in rms_alphas:
                freq_by_alpha[alpha] = (abs_res > alpha * layer_rms).mean(axis=0).astype(np.float64)
            freq_rms = freq_by_alpha.get(
                rms_alpha, (abs_res > rms_alpha * layer_rms).mean(axis=0).astype(np.float64)
            )
        else:  # no reservoir -> fall back to fixed-threshold frequency
            freq_rms = freq_fixed.copy()
            for alpha in rms_alphas:
                freq_by_alpha[alpha] = freq_fixed.copy()

        return LayerActivationResult(
            name=name,
            layer_index=layer_index,
            width=self.width,
            num_tokens=self.num_tokens,
            layer_rms=layer_rms,
            fixed_eps=self.fixed_eps,
            rms_alpha=rms_alpha,
            rms_eps=rms_alpha * layer_rms,
            mean_abs=mean_abs,
            rms=rms,
            freq_fixed=freq_fixed,
            freq_rms=freq_rms.astype(np.float64),
            reservoir_tokens=0 if reservoir is None else int(reservoir.shape[0]),
            rms_alphas=list(rms_alphas),
            freq_by_alpha=freq_by_alpha,
        )

    def reservoir_matrix(self) -> np.ndarray | None:
        if self.reservoir is None:
            return None
        return self.reservoir[: self._filled].to(torch.float32).numpy()


class NeuronActivationStats:
    """Manages per-layer accumulators; usable as a collector callback target."""

    def __init__(
        self,
        *,
        fixed_eps: float = 1e-3,
        rms_alpha: float = 1e-3,
        rms_alphas: Sequence[float] | None = None,
        max_reservoir_tokens: int = 4096,
        seed: int = 42,
    ) -> None:
        self.fixed_eps = fixed_eps
        self.rms_alpha = rms_alpha
        alphas = list(rms_alphas) if rms_alphas else list(_DEFAULT_ALPHAS)
        if rms_alpha not in alphas:
            alphas.append(rms_alpha)
        self.rms_alphas = sorted(set(alphas))
        self.max_reservoir_tokens = max_reservoir_tokens
        self._seed = seed
        self._accumulators: dict[str, _LayerAccumulator] = {}
        self._layer_index: dict[str, int] = {}

    def update(self, name: str, layer_index: int, act: torch.Tensor) -> None:
        acc = self._accumulators.get(name)
        if acc is None:
            acc = _LayerAccumulator(
                width=act.shape[-1],
                fixed_eps=self.fixed_eps,
                max_reservoir_tokens=self.max_reservoir_tokens,
                # Per-layer rng seeded by layer index for reproducible reservoirs.
                rng=np.random.default_rng(self._seed + max(layer_index, 0)),
            )
            self._accumulators[name] = acc
            self._layer_index[name] = layer_index
        acc.update(act)

    # Callback signature for NeuronActivationCollector.
    __call__ = update

    def names(self) -> list[str]:
        return sorted(self._accumulators, key=lambda n: (self._layer_index[n], n))

    def reservoir(self, name: str) -> np.ndarray | None:
        return self._accumulators[name].reservoir_matrix()

    def finalize(self) -> list[LayerActivationResult]:
        return [
            self._accumulators[name].finalize(
                name, self._layer_index[name], self.rms_alpha, self.rms_alphas
            )
            for name in self.names()
        ]
