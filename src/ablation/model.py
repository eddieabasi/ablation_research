"""A small, fully hookable transformer for ablation experiments.

Every intermediate we might want to ablate is passed through a named hook point.
A forward pass accepts:

* ``hooks``: ``{hook_name: fn}``. ``fn(tensor) -> tensor`` replaces the activation.
* ``cache``: a dict that is filled with ``{hook_name: tensor}`` for every hook point, or
  only those in ``cache_names`` if given (the value stored is the one the model actually
  used, i.e. after any hook ran).

Hook points (``l`` is the layer index):

* ``blocks.{l}.resid_pre``     [batch, pos, d_model]
* ``blocks.{l}.attn.pattern``  [batch, head, q_pos, k_pos]
* ``blocks.{l}.attn.result``   [batch, pos, head, d_model]   per-head output, after W_O
* ``blocks.{l}.mlp_out``       [batch, pos, d_model]         (only if the model has MLPs)
* ``resid_final``              [batch, pos, d_model]         before the final LayerNorm

Per-head ``attn.result`` is the natural unit for head ablation: the attention layer's
output is ``result.sum(head) + b_O``, so replacing one head's slice ablates exactly
that head and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

Hooks = dict[str, Callable[[torch.Tensor], torch.Tensor]]


@dataclass
class Config:
    vocab_size: int = 64
    n_ctx: int = 64
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 8
    d_head: int = 16
    attn_only: bool = True
    d_mlp: int = 256
    # If > 0, tokens[:, 0] is a task id in [0, n_tasks) and a learned task embedding is
    # added at every position, so no head has to move task identity around.
    n_tasks: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class _HookRunner:
    def __init__(self, hooks: Hooks | None, cache: dict | None, cache_names=None):
        self.hooks = hooks or {}
        self.cache = cache
        self.cache_names = set(cache_names) if cache_names is not None else None

    def __call__(self, name: str, x: torch.Tensor) -> torch.Tensor:
        fn = self.hooks.get(name)
        if fn is not None:
            x = fn(x)
        if self.cache is not None and (self.cache_names is None or name in self.cache_names):
            self.cache[name] = x
        return x


class Attention(nn.Module):
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        h, d, dh = cfg.n_heads, cfg.d_model, cfg.d_head
        self.W_Q = nn.Parameter(torch.randn(h, d, dh) / math.sqrt(d))
        self.W_K = nn.Parameter(torch.randn(h, d, dh) / math.sqrt(d))
        self.W_V = nn.Parameter(torch.randn(h, d, dh) / math.sqrt(d))
        self.W_O = nn.Parameter(torch.randn(h, dh, d) / math.sqrt(h * dh))
        self.b_Q = nn.Parameter(torch.zeros(h, dh))
        self.b_K = nn.Parameter(torch.zeros(h, dh))
        self.b_V = nn.Parameter(torch.zeros(h, dh))
        self.b_O = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor, hook: _HookRunner) -> torch.Tensor:
        p = f"blocks.{self.layer}.attn"
        q = torch.einsum("bpd,hde->bphe", x, self.W_Q) + self.b_Q
        k = torch.einsum("bpd,hde->bphe", x, self.W_K) + self.b_K
        v = torch.einsum("bpd,hde->bphe", x, self.W_V) + self.b_V
        scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / math.sqrt(self.cfg.d_head)
        n = x.shape[1]
        mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
        pattern = hook(f"{p}.pattern", scores.softmax(dim=-1))
        z = torch.einsum("bhqk,bkhe->bqhe", pattern, v)
        result = hook(f"{p}.result", torch.einsum("bqhe,hed->bqhd", z, self.W_O))
        return result.sum(dim=2) + self.b_O


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.fc_out = nn.Linear(cfg.d_mlp, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.gelu(self.fc_in(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        self.layer = layer
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg, layer)
        self.has_mlp = not cfg.attn_only
        if self.has_mlp:
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.mlp = MLP(cfg)

    def forward(self, resid: torch.Tensor, hook: _HookRunner) -> torch.Tensor:
        resid = hook(f"blocks.{self.layer}.resid_pre", resid)
        resid = resid + self.attn(self.ln1(resid), hook)
        if self.has_mlp:
            resid = resid + hook(f"blocks.{self.layer}.mlp_out", self.mlp(self.ln2(resid)))
        return resid


class Transformer(nn.Module):
    """Pre-LayerNorm decoder-only transformer with learned positional embeddings."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.task_embed = nn.Embedding(cfg.n_tasks, cfg.d_model) if cfg.n_tasks else None
        self.blocks = nn.ModuleList(Block(cfg, l) for l in range(cfg.n_layers))
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size)
        nn.init.normal_(self.embed.weight, std=0.5)
        nn.init.normal_(self.pos_embed.weight, std=0.5)
        if self.task_embed is not None:
            nn.init.normal_(self.task_embed.weight, std=0.5)

    def forward(
        self,
        tokens: torch.Tensor,
        hooks: Hooks | None = None,
        cache: dict | None = None,
        cache_names=None,
    ) -> torch.Tensor:
        hook = _HookRunner(hooks, cache, cache_names)
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        resid = self.embed(tokens) + self.pos_embed(pos)
        if self.task_embed is not None:
            resid = resid + self.task_embed(tokens[:, :1])
        for block in self.blocks:
            resid = block(resid, hook)
        resid = hook("resid_final", resid)
        return self.unembed_resid(resid)

    def unembed_resid(self, resid_final: torch.Tensor) -> torch.Tensor:
        """Final LayerNorm + unembedding. Exposed so path-specific ablations can reuse it."""
        return self.unembed(self.ln_final(resid_final))

    @torch.no_grad()
    def run_with_cache(self, tokens: torch.Tensor, hooks: Hooks | None = None, names=None):
        """Forward pass returning (logits, cache). ``names`` limits what is cached (saves memory)."""
        cache: dict[str, torch.Tensor] = {}
        logits = self(tokens, hooks=hooks, cache=cache, cache_names=names)
        return logits, cache

    def save(self, path) -> None:
        torch.save({"cfg": self.cfg.to_dict(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path) -> "Transformer":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(Config(**blob["cfg"]))
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model
