"""
Forward hooks for collecting calibration statistics used by the Wanda and
SparseGPT tile-importance metrics.

For each target linear module we accumulate the input Gram matrix

    H = sum_over_calibration_tokens  X^T X        (shape [in_features, in_features])

where X is the module input. A single statistic serves both metrics:

  * Wanda      needs the per-input-column activation norm
               ||X_:,j||_2 = sqrt(H[j, j]).
  * SparseGPT  (masked-tile error) needs the full H to measure the relative
               change in the layer output caused by removing a tile.

H is accumulated unnormalised. Both metrics use it in a scale-invariant way
(Wanda takes a column norm, SparseGPT a ratio of quadratic forms), so the
calibration token count cancels out and no normalisation is required.
"""

import torch


def _flatten_input(x):
    # (batch, seq, in_features) or any (..., in_features) -> (n_tokens, in_features)
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    return x


class GramCollector:
    """Accumulates H = sum X^T X for one linear module via a forward pre-hook."""

    def __init__(self, module, name):
        self.name = name
        in_features = module.weight.shape[1]
        device = module.weight.device
        self.H = torch.zeros(in_features, in_features, dtype=torch.float32, device=device)
        self.n_tokens = 0
        self._handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        x = _flatten_input(args[0].detach()).to(torch.float32)
        self.H += x.t() @ x
        self.n_tokens += x.shape[0]

    def col_norms(self):
        """Wanda per-column activation norms ||X_:,j||_2 = sqrt(diag(H))."""
        return torch.sqrt(torch.clamp(torch.diag(self.H), min=0.0))

    def remove(self):
        self._handle.remove()


@torch.no_grad()
def collect_gram_stats(model, modules_by_name, calib_samples, device=None):
    """Run the calibration samples through the model once and return
    {name: GramCollector} for every target module in modules_by_name.

    modules_by_name : dict {target_name: nn.Linear}
    calib_samples   : list of input_id tensors, each shape [1, seqlen]
    """
    if device is None:
        device = model.device

    collectors = {name: GramCollector(mod, name) for name, mod in modules_by_name.items()}
    try:
        for input_ids in calib_samples:
            model(input_ids.to(device))
    finally:
        for c in collectors.values():
            c.remove()

    return collectors
