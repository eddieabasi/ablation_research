"""Training loop for toy models. ``task`` is anything with ``sample(batch_size, generator) -> Batch``."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .model import Config, Transformer
from . import tasks as T


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup: int = 100
    seed: int = 0
    log_every: int = 250


def train(
    model_cfg: Config,
    task,
    train_cfg: TrainConfig,
    verbose: bool = True,
) -> tuple[Transformer, list[dict]]:
    torch.manual_seed(train_cfg.seed)
    gen = torch.Generator().manual_seed(10_000 + train_cfg.seed)
    model = Transformer(model_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / train_cfg.warmup)
    )
    log: list[dict] = []
    t0 = time.time()
    model.train()
    for step in range(train_cfg.steps):
        batch = task.sample(train_cfg.batch_size, gen)
        loss = T.loss(model(batch.tokens), batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % train_cfg.log_every == 0 or step == train_cfg.steps - 1:
            entry = {"step": step, "loss": loss.item(), "elapsed_s": round(time.time() - t0, 1)}
            log.append(entry)
            if verbose:
                print(f"  seed={train_cfg.seed} step={step:5d} loss={loss.item():.4f}")
    model.eval()
    return model, log
