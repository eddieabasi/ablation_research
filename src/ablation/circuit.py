"""Ablation-free identification of the reference induction circuit.

The point of the toy setting is to have a notion of "which heads matter" that does not
come from any ablation method, so ablation methods can be scored against it. Here that
reference comes from attention patterns, which ablation does not touch:

* previous-token score: attention from position t to t-1
* two-back score: attention from t to t-2 (the two-back task's head)
* induction score: attention from an induction position t to keys s whose *previous*
  token equals the current token (``tokens[s-1] == tokens[t]``), i.e. the position
  holding the answer
* duplicate-token score: attention to earlier occurrences of the current token itself
* BOS score: attention to position 0 (the usual "resting" behaviour of idle heads)

A head is in the reference circuit if it is a layer-0 previous-token head or a later
induction head (score above ``threshold``). Attention patterns say *where* a head looks,
not whether its output is used, so this is the reference used for the "textbook"
mechanism; the experiments also check it is causally sufficient (see ``sufficiency``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import Transformer
from .tasks import Batch


@dataclass
class HeadScores:
    prev_token: torch.Tensor  # [layer, head]
    two_back: torch.Tensor
    induction: torch.Tensor
    duplicate: torch.Tensor
    bos: torch.Tensor

    def to_dict(self) -> dict:
        return {k: v.tolist() for k, v in self.__dict__.items()}


@torch.no_grad()
def head_scores(model: Transformer, batch: Batch) -> HeadScores:
    names = [f"blocks.{l}.attn.pattern" for l in range(model.cfg.n_layers)]
    _, cache = model.run_with_cache(batch.tokens, names=names)
    tok = batch.tokens
    b, n = tok.shape
    # key s is an "answer" key for query t if tokens[s-1] == tokens[t], 1 <= s <= t
    prev_of_key = torch.cat([torch.full((b, 1), -1), tok[:, :-1]], dim=1)  # [b, s]
    ans = prev_of_key.unsqueeze(1) == tok.unsqueeze(2)  # [b, t, s]
    dup = tok.unsqueeze(1) == tok.unsqueeze(2)
    causal = torch.ones(n, n, dtype=torch.bool).tril(-1)  # s < t (excludes self)
    dup = dup & causal
    ans = ans & torch.ones(n, n, dtype=torch.bool).tril(0)
    q_mask = batch.mask.float()  # induction query positions
    prev_mask = torch.zeros(b, n)
    prev_mask[:, 2:] = 1.0  # queries whose previous token is not BOS

    L, H = model.cfg.n_layers, model.cfg.n_heads
    out = {k: torch.zeros(L, H) for k in ("prev_token", "two_back", "induction", "duplicate", "bos")}
    idx = torch.arange(1, n)
    for l in range(L):
        pat = cache[f"blocks.{l}.attn.pattern"]  # [b, h, t, s]
        prev = torch.zeros(b, H, n)
        prev[:, :, 1:] = pat[:, :, idx, idx - 1]
        out["prev_token"][l] = _masked_mean(prev, prev_mask)
        back2 = torch.zeros(b, H, n)
        back2[:, :, 2:] = pat[:, :, idx[1:], idx[1:] - 2]
        out["two_back"][l] = _masked_mean(back2, prev_mask)
        out["induction"][l] = _masked_mean((pat * ans.unsqueeze(1)).sum(-1), q_mask)
        out["duplicate"][l] = _masked_mean((pat * dup.unsqueeze(1)).sum(-1), q_mask)
        out["bos"][l] = _masked_mean(pat[..., 0], q_mask)
    return HeadScores(**out)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x [b, h, t], mask [b, t] -> [h]"""
    m = mask.unsqueeze(1)
    return (x * m).sum((0, 2)) / m.sum((0, 2))


def reference_circuit(scores: HeadScores, threshold: float = 0.3) -> set[tuple[int, int]]:
    """Layer-0 previous-token heads and layer>=1 induction heads."""
    heads: set[tuple[int, int]] = set()
    L, H = scores.prev_token.shape
    for l in range(L):
        for h in range(H):
            if l == 0 and scores.prev_token[l, h] > threshold:
                heads.add((l, h))
            if l >= 1 and scores.induction[l, h] > threshold:
                heads.add((l, h))
    return heads
