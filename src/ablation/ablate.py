"""Ablation methods for attention heads.

Each method replaces the per-head output ``blocks.{l}.attn.result[:, :, h]`` (which is
exactly that head's additive contribution to the residual stream) with something else:

* ``zero``      0
* ``mean``      the head's mean output at that position over a reference set of inputs
* ``resample``  the head's output on a *different* input drawn from the same distribution
                (each evaluation sequence is paired with one reference sequence)
* ``optimal``   the constant (per position) that minimises expected loss, fitted by
                gradient descent on held-out training samples, starting from the mean
                (Li & Janson, 2024, "Optimal ablation for interpretability")

The *effect* of an ablation is the per-sequence increase in task loss:
``loss(ablated) - loss(clean)``.

What each method removes, for intuition:

* ``optimal`` removes only the input-dependent information the head carries, and lets
  the rest of the model see the best possible stand-in. It is the smallest of the
  constant-replacement effects (up to fitting noise), since ``mean`` and ``zero`` are
  also constants.
* ``mean`` removes the information and uses a typical-looking stand-in.
* ``zero`` removes the information *and* the head's mean output. If the mean is large,
  the residual stream is pushed somewhere it never is during training.
* ``resample`` replaces the information with *wrong* information that looks
  in-distribution, which downstream heads may act on confidently.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch

from . import tasks as T
from .model import Hooks, Transformer

METHODS = ("zero", "mean", "resample", "optimal")
Head = tuple[int, int]


def result_hook_name(layer: int) -> str:
    return f"blocks.{layer}.attn.result"


@dataclass
class AblationContext:
    """Everything needed to ablate heads of one model on one evaluation set.

    ``eval_batch``     sequences the effect is measured on
    ``ref_batch``      sequences the per-position mean is computed from
    ``resample_batch`` sequences supplying replacement activations (same size as eval)
    ``sampler``        draws fresh training-distribution batches, used to fit optimal constants
    """

    model: Transformer
    eval_batch: T.Batch
    ref_batch: T.Batch
    resample_batch: T.Batch
    sampler: Callable[[int], T.Batch]
    clean_loss: torch.Tensor = field(init=False)  # per sequence
    mean_result: dict[int, torch.Tensor] = field(init=False)  # layer -> [pos, head, d]
    resample_result: dict[int, torch.Tensor] = field(init=False)  # layer -> [b, pos, head, d]

    def __post_init__(self) -> None:
        if len(self.resample_batch) != len(self.eval_batch):
            raise ValueError("resample_batch must be the same size as eval_batch")
        L = self.model.cfg.n_layers
        with torch.no_grad():
            self.clean_loss = T.per_seq_loss(self.model(self.eval_batch.tokens), self.eval_batch)
            names = [result_hook_name(l) for l in range(L)]
            _, ref = self.model.run_with_cache(self.ref_batch.tokens, names=names)
            _, res = self.model.run_with_cache(self.resample_batch.tokens, names=names)
        self.mean_result = {l: ref[result_hook_name(l)].mean(0) for l in range(L)}
        self.resample_result = {l: res[result_hook_name(l)] for l in range(L)}

    @property
    def all_heads(self) -> list[Head]:
        cfg = self.model.cfg
        return [(l, h) for l in range(cfg.n_layers) for h in range(cfg.n_heads)]


def _by_layer(heads: Iterable[Head]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for l, h in heads:
        out[l].append(h)
    return dict(out)


def ablation_hooks(
    ctx: AblationContext,
    heads: Iterable[Head],
    method: str,
    constants: dict[int, torch.Tensor] | None = None,
) -> Hooks:
    """Hooks replacing the outputs of ``heads``.

    For ``optimal``, ``constants[layer]`` is a [pos, n_selected_heads, d] tensor in the
    order the heads of that layer appear in ``heads``.
    """
    hooks: Hooks = {}
    for layer, hs in _by_layer(heads).items():
        idx = torch.tensor(hs)
        if method == "zero":
            repl = lambda x, idx=idx: torch.zeros_like(x[:, :, idx])
        elif method == "mean":
            m = ctx.mean_result[layer][:, idx]
            repl = lambda x, m=m: m.expand(x.shape[0], *m.shape)
        elif method == "resample":
            r = ctx.resample_result[layer][:, :, idx]
            repl = lambda x, r=r: r[: x.shape[0]]
        elif method == "optimal":
            if constants is None:
                raise ValueError("optimal ablation needs fitted constants")
            c = constants[layer]
            repl = lambda x, c=c: c.expand(x.shape[0], *c.shape)
        else:
            raise ValueError(f"unknown method {method!r}")

        def fn(x: torch.Tensor, idx=idx, repl=repl) -> torch.Tensor:
            x = x.clone()
            x[:, :, idx] = repl(x)
            return x

        hooks[result_hook_name(layer)] = fn
    return hooks


def fit_optimal_constants(
    ctx: AblationContext,
    heads: Iterable[Head],
    steps: int = 300,
    batch_size: int = 128,
    lr: float = 0.03,
) -> dict[int, torch.Tensor]:
    """Fit per-position constants minimising training-distribution loss, init at the mean."""
    heads = list(heads)
    params = {
        l: ctx.mean_result[l][:, torch.tensor(hs)].clone().requires_grad_(True)
        for l, hs in _by_layer(heads).items()
    }
    model = ctx.model
    req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(params.values(), lr=lr)
    try:
        for _ in range(steps):
            batch = ctx.sampler(batch_size)
            hooks = ablation_hooks(ctx, heads, "optimal", constants=params)
            loss = T.loss(model(batch.tokens, hooks=hooks), batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    finally:
        for p, r in zip(model.parameters(), req):
            p.requires_grad_(r)
    return {l: p.detach() for l, p in params.items()}


@torch.no_grad()
def ablation_effect(
    ctx: AblationContext,
    heads: Iterable[Head],
    method: str,
    constants: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Per-sequence loss increase on ``ctx.eval_batch`` from ablating ``heads`` jointly."""
    heads = list(heads)
    if method == "optimal" and constants is None:
        with torch.enable_grad():
            constants = fit_optimal_constants(ctx, heads)
    hooks = ablation_hooks(ctx, heads, method, constants)
    logits = ctx.model(ctx.eval_batch.tokens, hooks=hooks)
    return T.per_seq_loss(logits, ctx.eval_batch) - ctx.clean_loss
