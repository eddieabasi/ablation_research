"""Small statistics helpers (bootstrap CIs, rank agreement, AUROC)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI of the mean."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def spearman(a, b) -> float:
    return float(stats.spearmanr(a, b).statistic)


def auroc(scores, labels) -> float:
    """P(score of a random positive > score of a random negative), ties count half."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def top_k_overlap(a, b, k: int) -> float:
    """|top-k(a) ∩ top-k(b)| / k."""
    ta = set(np.argsort(-np.asarray(a))[:k])
    tb = set(np.argsort(-np.asarray(b))[:k])
    return len(ta & tb) / k


def interaction_test(m0, mA, mB, mAB, mask) -> dict:
    """Is an interaction between two ablations A and B created by the loss curvature?

    Inputs are per-position margins [batch, seq] (torch tensors) for clean (m0), A alone,
    B alone, and both (mAB), plus the mask of scored positions. Returns:

    * ``loss_*``: loss increases (nats) for A, B, AB and their interaction
      ``loss_AB - loss_A - loss_B``
    * ``margin_*``: mean margin *drops* and their interaction (positive = superadditive)
    * ``pred_interaction``: the loss interaction implied if margin drops added exactly,
      i.e. using margins mA + mB - m0 for AB. ``explained = pred / actual``.
    """
    import torch.nn.functional as F

    w = mask.float()
    per_seq = lambda x: ((x * w).sum(-1) / w.sum(-1)).mean()  # same weighting as tasks.per_seq_loss
    L = lambda m: float(per_seq(F.softplus(-m)))
    M = lambda m: float(per_seq(m0 - m))
    l0 = L(m0)
    la, lb, lab = L(mA) - l0, L(mB) - l0, L(mAB) - l0
    lpred = L(mA + mB - m0) - l0
    actual = lab - la - lb
    pred = lpred - la - lb
    ma, mb, mab = M(mA), M(mB), M(mAB)
    return {
        "loss_A": la, "loss_B": lb, "loss_AB": lab, "loss_interaction": actual,
        "margin_drop_A": ma, "margin_drop_B": mb, "margin_drop_AB": mab,
        "margin_interaction": mab - ma - mb,
        "pred_interaction": pred,
        "explained": pred / actual if abs(actual) > 1e-9 else float("nan"),
    }
