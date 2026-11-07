"""Tasks with a known mechanism, so ablation conclusions can be checked against it.

Every task produces a :class:`Batch` of ``tokens``, per-position ``targets`` and a ``mask``
of positions where the loss is defined. The model's output at position ``t`` is scored
against ``targets[t]``. Position 0 holds a task tag (a BOS token that differs per task),
so a model trained on a :class:`Mixture` can tell tasks apart.

Vocabulary: ids ``0..n_tags-1`` are task tags, the rest are content tokens.

Induction task (tag 0)
----------------------
Tag followed by a *periodic* run of content tokens: a block of ``D`` distinct tokens
``r_1..r_D`` repeated until the sequence is full, with the period ``D`` drawn per sequence.

    position:  0    1 .. D         D+1 .. 2D        2D+1 ...
    token:     TAG  r_1 .. r_D     r_1 .. r_D       r_1 ...

At ``t >= D+1`` the current token occurred before and the target (next token) is whatever
followed that earlier occurrence. Those are the masked-in *induction positions*.

The period is random because with a fixed period the answer always sits a fixed distance
back and a model with learned positional embeddings solves the task with one positional
"copy from t-D+1" head and no induction (we saw exactly this shortcut in a first version:
loss collapsed within ~250 steps with no plateau). A random ``D`` forces content lookup.

The textbook mechanism in a 2-layer attention-only model (Olsson et al., 2022) is a layer-0
*previous-token head* (writes "my previous token was X" at each position) feeding, via
K-composition, a layer-1 *induction head* (attends to positions whose previous token equals
the current token, and copies the token found there).

Two-back task (tag 1)
---------------------
Tag followed by i.i.d. content tokens; the target at position ``t`` is ``tokens[t-2]``.
It needs a head that attends two positions back and copies, a different circuit from
induction. In a model trained on both tasks, heads serving two-back are *active but
irrelevant* on induction inputs: exactly the kind of component an ablation method should
not flag as important for induction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

N_TAGS = 2
TAG_INDUCTION = 0
TAG_TWO_BACK = 1


@dataclass
class Batch:
    tokens: torch.Tensor  # [batch, seq] long
    targets: torch.Tensor  # [batch, seq] long (-1 where undefined)
    mask: torch.Tensor  # [batch, seq] bool, True where the loss is defined

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx) -> "Batch":
        return Batch(self.tokens[idx], self.targets[idx], self.mask[idx])

    @staticmethod
    def cat(batches: list["Batch"]) -> "Batch":
        return Batch(*(torch.cat([getattr(b, k) for b in batches]) for k in ("tokens", "targets", "mask")))


def per_seq_loss(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Mean cross-entropy over masked positions, one value per sequence."""
    lp = logits.log_softmax(dim=-1)
    nll = -lp.gather(-1, batch.targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    m = batch.mask.float()
    return (nll * m).sum(-1) / m.sum(-1)


def loss(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    return per_seq_loss(logits, batch).mean()


def accuracy(logits: torch.Tensor, batch: Batch) -> float:
    correct = (logits.argmax(-1) == batch.targets) & batch.mask
    return (correct.sum() / batch.mask.sum()).item()


@dataclass(frozen=True)
class InductionTask:
    vocab_size: int = 64
    seq_len: int = 56  # including the tag
    min_period: int = 8
    max_period: int = 24
    tag: int = TAG_INDUCTION

    @property
    def n_content(self) -> int:
        return self.vocab_size - N_TAGS

    @property
    def chance_loss(self) -> float:
        """Loss of a model that knows only the content alphabet (uniform prediction)."""
        return math.log(self.n_content)

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> Batch:
        if self.max_period > self.n_content:
            raise ValueError("max_period must not exceed the number of content tokens")
        if self.seq_len < self.max_period + 3:
            raise ValueError("seq_len too short to contain an induction position for every period")
        period = torch.randint(self.min_period, self.max_period + 1, (batch_size,), generator=generator)
        # Distinct tokens within a period, so "the earlier occurrence" is unambiguous.
        block = torch.rand(batch_size, self.n_content, generator=generator).argsort(dim=-1) + N_TAGS
        idx = torch.arange(self.seq_len - 1).unsqueeze(0) % period.unsqueeze(1)
        body = block.gather(1, idx)
        tokens = torch.cat([torch.full((batch_size, 1), self.tag, dtype=torch.long), body], dim=1)

        targets = torch.full_like(tokens, -1)
        targets[:, :-1] = tokens[:, 1:]
        t = torch.arange(self.seq_len).unsqueeze(0)
        mask = (t >= period.unsqueeze(1) + 1) & (t <= self.seq_len - 2)
        return Batch(tokens, targets, mask)


@dataclass(frozen=True)
class TwoBackTask:
    vocab_size: int = 64
    seq_len: int = 56
    lag: int = 2
    tag: int = TAG_TWO_BACK

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> Batch:
        body = torch.randint(N_TAGS, self.vocab_size, (batch_size, self.seq_len - 1), generator=generator)
        tokens = torch.cat([torch.full((batch_size, 1), self.tag, dtype=torch.long), body], dim=1)
        targets = torch.full_like(tokens, -1)
        targets[:, 1 + self.lag :] = tokens[:, 1 : -self.lag]
        mask = targets >= 0
        return Batch(tokens, targets, mask)


@dataclass(frozen=True)
class Mixture:
    tasks: tuple = field(default_factory=lambda: (InductionTask(), TwoBackTask()))
    weights: tuple = (0.5, 0.5)

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> Batch:
        counts = torch.multinomial(torch.tensor(self.weights), batch_size, replacement=True, generator=generator)
        parts = [t.sample(int((counts == i).sum()), generator) for i, t in enumerate(self.tasks)]
        parts = [p for p in parts if len(p)]
        return Batch.cat(parts)
