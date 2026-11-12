"""Exact decomposition of an ablation's effect on the target logit (Q3: self-repair).

Target logit difference ``ld = z_target - mean_v z_v`` at every position where the loss
is defined. With the final LayerNorm's scale held at its clean value ``s``, ``ld`` is a
linear function of the final residual stream ``r``::

    ld = ((r - mean(r)) / s * gamma + beta) . u + const,   u = W_U[target] - mean_v W_U[v]

and ``r`` is the sum of the embeddings and every head's output. So ``ld`` splits into one
contribution per component, ``c(x) = ((x - mean(x)) / s * gamma) . u``, plus a constant.

Ablating a set of heads changes the actual ``ld`` by exactly::

    total = direct + sum(downstream responses) + final-LayerNorm term

* ``direct``     the ablated heads' own change in contribution (replacement - clean)
* ``responses``  for every head in a later layer, the change in its contribution, because
                 its inputs changed. Negative responses add damage (the head relied on the
                 ablated one); positive responses repair it (self-repair, "Hydra effect").
* ``ln_final``   the rest: the final LayerNorm rescaling the changed residual stream.

Heads in the same or earlier layers are unaffected, so their change is exactly 0. The
embeddings never change. All quantities are means over each sequence's scored positions,
in logit units (nats of target-logit advantage).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import tasks as T
from .ablate import AblationContext, Head, ablation_hooks, frozen_ln_hooks, result_hook_name


def logit_diff(logits: torch.Tensor, batch: T.Batch) -> torch.Tensor:
    """Per-position target logit minus mean logit, [batch, seq]."""
    z_t = logits.gather(-1, batch.targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return z_t - logits.mean(-1)


def seq_mean(x: torch.Tensor, batch: T.Batch) -> torch.Tensor:
    """Mean over scored positions, one value per sequence."""
    m = batch.mask.float()
    return (x * m).sum(-1) / m.sum(-1)


@dataclass
class Decomposition:
    total: torch.Tensor  # [b] change in logit diff (actual forward pass)
    direct: torch.Tensor  # [b] ablated heads' own change in contribution
    responses: dict[Head, torch.Tensor]  # head in a later layer -> [b] change in its contribution
    ln_final: torch.Tensor  # [b] total - (direct + sum of responses)
    loss: torch.Tensor  # [b] loss increase
    margin_drop: torch.Tensor  # [b] mean drop of the margin z_t - logsumexp(others)

    @property
    def response_sum(self) -> torch.Tensor:
        return sum(self.responses.values(), torch.zeros_like(self.total))


class Decomposer:
    """Caches clean-run quantities for one :class:`AblationContext`."""

    def __init__(self, ctx: AblationContext):
        self.ctx = ctx
        model, batch = ctx.model, ctx.eval_batch
        with torch.no_grad():
            logits = model(batch.tokens)
            W = model.unembed.weight  # [vocab, d]
            self.u = W[batch.targets.clamp(min=0)] - W.mean(0)  # [b, t, d]
            self.scale = ctx.clean_ln_scales["ln_final.scale"]  # [b, t, 1]
            self.ld_clean = seq_mean(logit_diff(logits, batch), batch)
            self.m_clean = T.margins(logits, batch)
            self.c_clean = self.contributions(ctx.clean_result)

    @torch.no_grad()
    def contributions(self, results: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Per-head contribution to the logit diff (final LayerNorm scale frozen at clean), layer -> [b, heads]."""
        g = self.ctx.model.ln_final.weight
        out = {}
        for l, r in results.items():  # r [b, t, h, d]
            rc = (r - r.mean(-1, keepdim=True)) * g / self.scale.unsqueeze(2)
            c = torch.einsum("bthd,btd->bth", rc, self.u)
            m = self.ctx.eval_batch.mask.float().unsqueeze(-1)
            out[l] = (c * m).sum(1) / m.sum(1)
        return out

    @torch.no_grad()
    def __call__(
        self,
        heads: list[Head],
        method: str,
        constants: dict[int, torch.Tensor] | None = None,
        freeze_ln: str | None = None,
    ) -> Decomposition:
        """Decompose the effect of ablating ``heads`` jointly.

        ``freeze_ln``: optionally hold some LayerNorm scales at their clean values during the
        ablated run (``"attn_inputs"``: the LayerNorms in front of attention layers, so
        downstream heads cannot be rescaled; ``"final"``: then ``ln_final`` is 0).
        """
        ctx, model, batch = self.ctx, self.ctx.model, self.ctx.eval_batch
        L = model.cfg.n_layers
        hooks = ablation_hooks(ctx, heads, method, constants)
        if freeze_ln:
            hooks.update(frozen_ln_hooks(ctx, freeze_ln))
        names = [result_hook_name(l) for l in range(L)]
        logits, cache = model.run_with_cache(batch.tokens, hooks=hooks, names=names)
        c_abl = self.contributions({l: cache[result_hook_name(l)] for l in range(L)})
        delta = {l: c_abl[l] - self.c_clean[l] for l in range(L)}
        ablated = set(heads)
        first = min(l for l, _ in heads)
        direct = sum((delta[l][:, h] for l, h in heads), torch.zeros(len(batch)))
        responses = {
            (l, h): delta[l][:, h]
            for l in range(first + 1, L)
            for h in range(model.cfg.n_heads)
            if (l, h) not in ablated
        }
        total = seq_mean(logit_diff(logits, batch), batch) - self.ld_clean
        frozen_total = sum((d.sum(-1) for d in delta.values()), torch.zeros(len(batch)))
        return Decomposition(
            total=total,
            direct=direct,
            responses=responses,
            ln_final=total - frozen_total,
            loss=T.per_seq_loss(logits, batch) - ctx.clean_loss,
            margin_drop=seq_mean(self.m_clean - T.margins(logits, batch), batch),
        )

    @torch.no_grad()
    def unaffected_change(self, heads: list[Head], method: str) -> float:
        """Largest change in contribution of any head that cannot be affected (same or earlier
        layer, not ablated). Should be 0 up to float error; used as a self-check."""
        ctx, model, batch = self.ctx, self.ctx.model, self.ctx.eval_batch
        L = model.cfg.n_layers
        names = [result_hook_name(l) for l in range(L)]
        _, cache = model.run_with_cache(batch.tokens, hooks=ablation_hooks(ctx, heads, method), names=names)
        c_abl = self.contributions({l: cache[result_hook_name(l)] for l in range(L)})
        first = min(l for l, _ in heads)
        worst = 0.0
        for l in range(first + 1):
            for h in range(model.cfg.n_heads):
                if (l, h) not in set(heads):
                    worst = max(worst, (c_abl[l][:, h] - self.c_clean[l][:, h]).abs().max().item())
        return worst
