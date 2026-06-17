"""Synthesize a single EMA model checkpoint from per-step EMA snapshots.

Algorithm 3 (post-hoc EMA) from https://arxiv.org/abs/2312.02696, implemented
in `nitrous_ema.PostHocEMA.synthesize_ema_model`. Given checkpoints saved at
`(sigma_rel_i, step_i)` pairs, return weights for a target `(sigma_rel, step)`.

Bug fix (2026-05-18): when training compiles the model via
`torch.compile(self.network.module)` (see runner.py:94 — perf fix for Step 3),
EMA snapshots saved by nitrous_ema carry keys of the form
`ema_model._orig_mod.<param>` instead of `ema_model.<param>`. The fresh
KarrasEMA instance built inside `synthesize_ema_model` wraps an *uncompiled*
model, so its `load_state_dict(..., strict=True)` rejects the prefixed keys.
Symptom: `synthesize_ema` raised `RuntimeError: Missing key(s) in state_dict`,
the caller in `train.py` swallowed it with a `WARNING`, and `ema_final.pth`
was never written — `sample()` then ran on randomly-initialized weights and
silently produced garbage predictions. We monkey-patch `torch.load` for the
synthesis call only, stripping the `._orig_mod.` segment on the fly. Snapshot
files on disk are left untouched, so the fix is reversible.
"""
import torch
from nitrous_ema import PostHocEMA
from omegaconf import DictConfig

from src.model.networks import get_model


def _strip_orig_mod(state_dict):
    """Remove the `._orig_mod.` segment that torch.compile inserts into nested module keys."""
    if not isinstance(state_dict, dict):
        return state_dict
    if not any("_orig_mod" in k for k in state_dict.keys()):
        return state_dict
    return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}


def synthesize_ema(cfg: DictConfig, sigma: float, step: int | None):
    model = get_model(cfg.variant, cfg.sample_rate)
    emas = PostHocEMA(
        model,
        sigma_rels=cfg.ema.sigma_rels,
        update_every=cfg.ema.update_every,
        checkpoint_every_num_steps=cfg.ema.checkpoint_every,
        checkpoint_folder=cfg.ema.checkpoint_folder,
    )

    orig_torch_load = torch.load

    def patched_load(*args, **kwargs):
        return _strip_orig_mod(orig_torch_load(*args, **kwargs))

    torch.load = patched_load
    try:
        synthesized_ema = emas.synthesize_ema_model(
            sigma_rel=sigma, step=step, device="cpu"
        )
    finally:
        torch.load = orig_torch_load

    state_dict = synthesized_ema.ema_model.state_dict()
    return state_dict
