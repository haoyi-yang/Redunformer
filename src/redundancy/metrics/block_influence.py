"""Block Influence (BI) redundancy measurement.

Definition (Men et al., 2024, ShortGPT): a block is redundant if its output hidden
state is nearly identical to its input, so BI_i = 1 - mean_t cos_sim(X_i,t, X_{i+1,t}).
GPT-2's forward pass with output_hidden_states=True already returns each block's
input/output as consecutive entries in `hidden_states`, so no custom hooks are needed.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class LayerStats:
    """Per-layer statistics accumulated over a calibration set of windows."""

    bi_scores: np.ndarray               # (n_layers,) -- 1 - mean cosine similarity of block input/output
    relative_residual_norms: np.ndarray  # (n_layers,) -- mean_t (||X_{i+1,t} - X_i,t|| / ||X_i,t||)
    layer_means: np.ndarray             # (n_layers + 1, hidden_dim) -- mean-pooled representation per layer
    n_layers: int
    n_windows: int
    n_tokens: int


@torch.no_grad()
def collect_layer_stats(
    model, input_windows: torch.Tensor, device, max_windows: int | None = None
) -> LayerStats:
    """Single forward pass per calibration window; accumulates running per-layer statistics.

    Processes windows one at a time (mirrors eval.compute_perplexity's loop) so memory stays
    bounded regardless of calibration-set size: only reduced running sums are kept, never the
    full stack of per-token hidden states.
    """
    n_windows = input_windows.size(0)
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    n_layers = None
    cos_sum = norm_sum = mean_sum = None
    n_tokens = 0

    for i in range(n_windows):
        ids = input_windows[i: i + 1].to(device)
        outputs = model(ids, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim), len = n_layers + 1

        if n_layers is None:
            n_layers = len(hidden_states) - 1
            hidden_dim = hidden_states[0].size(-1)
            cos_sum = torch.zeros(n_layers, dtype=torch.float64)
            norm_sum = torch.zeros(n_layers, dtype=torch.float64)
            mean_sum = torch.zeros(n_layers + 1, hidden_dim, dtype=torch.float64)

        n_tokens += ids.size(1)

        # Reduce on-device in float32 first (MPS has no float64 support), then move the small
        # reduced tensors to CPU and upcast for accumulation.
        for layer_idx, hs in enumerate(hidden_states):
            mean_sum[layer_idx] += hs[0].float().sum(dim=0).cpu().double()

        for block_idx in range(n_layers):
            x_in = hidden_states[block_idx][0].float()
            x_out = hidden_states[block_idx + 1][0].float()
            cos_sum[block_idx] += F.cosine_similarity(x_in, x_out, dim=-1).sum().cpu().double()
            # Relative update: ||X_out - X_in|| / ||X_in|| per token. Normalizing by the input
            # magnitude makes the update comparable across depth (the raw residual-stream norm
            # grows with depth regardless of redundancy). eps guards the rare zero-norm token.
            rel = (x_out - x_in).norm(dim=-1) / x_in.norm(dim=-1).clamp_min(1e-6)
            norm_sum[block_idx] += rel.sum().cpu().double()

        if (i + 1) % 50 == 0 or (i + 1) == n_windows:
            print(f"  [{i + 1}/{n_windows}] windows processed")

    return LayerStats(
        bi_scores=(1.0 - cos_sum / n_tokens).numpy(),
        relative_residual_norms=(norm_sum / n_tokens).numpy(),
        layer_means=(mean_sum / n_tokens).numpy(),
        n_layers=n_layers,
        n_windows=n_windows,
        n_tokens=n_tokens,
    )


def compute_cosine_similarity_matrix(layer_means: np.ndarray, center: bool = True) -> np.ndarray:
    """(L+1)x(L+1) cosine similarity matrix between per-layer representations.

    Reveals structure beyond adjacent layers (e.g. a block behaving like a near-duplicate of
    a non-neighbouring block), complementing the adjacent-only BI score.

    GPT-2's residual stream shares a large common direction (plus known outlier dimensions), so
    the raw uncentered cosine between layer means saturates near 1.0 almost everywhere and the
    heatmap washes out -- an artifact of the shared component, not evidence of redundancy. By
    default we center the layer means (subtract their mean across layers) before taking the
    cosine, which removes that shared component so real structure is visible. This is a cheap
    stand-in for linear CKA, which centers activations for the same reason.
    """
    x = layer_means - layer_means.mean(axis=0, keepdims=True) if center else layer_means
    return cosine_similarity(x)


@torch.no_grad()
def compute_tokenwise_cosine_similarity_matrix(
    model, input_windows: torch.Tensor, device, max_windows: int | None = None
) -> np.ndarray:
    """Average tokenwise cosine similarity for every pair of hidden states.

    Each token vector is normalized independently before pairwise similarities are
    accumulated. This makes adjacent entries consistent with Block Influence
    (similarity = 1 - BI) and avoids the artifact caused by centering layer means
    across depth.
    """
    n_windows = input_windows.size(0)
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    similarity_sum = None
    n_tokens = 0

    for i in range(n_windows):
        ids = input_windows[i : i + 1].to(device)
        hidden_states = model(
            ids, output_hidden_states=True, use_cache=False
        ).hidden_states

        normalized = torch.stack(
            [F.normalize(hidden[0].float(), dim=-1) for hidden in hidden_states]
        )
        flattened = normalized.flatten(start_dim=1)
        window_sum = flattened @ flattened.T

        if similarity_sum is None:
            similarity_sum = torch.zeros_like(window_sum, device="cpu", dtype=torch.float64)
        similarity_sum += window_sum.cpu().double()
        n_tokens += ids.size(1)

        if (i + 1) % 50 == 0 or (i + 1) == n_windows:
            print(f"  [{i + 1}/{n_windows}] similarity windows processed")

    matrix = (similarity_sum / n_tokens).numpy()
    # A vector compared with itself is exactly 1. TF32 matrix multiplication can
    # leave the diagonal a few 1e-4 below 1 through rounding.
    np.fill_diagonal(matrix, 1.0)
    return matrix
