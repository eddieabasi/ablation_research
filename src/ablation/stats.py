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
