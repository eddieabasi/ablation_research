"""Aggregate Q1 results across seeds into tables (summary.md / summary.json) and figures.

    python experiments/q1_analyze.py results/q1
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ablation.ablate import METHODS
from ablation.stats import auroc, spearman, top_k_overlap

# Reference palette, categorical slots 1-4 (validated light-mode, adjacent order),
# with marker shapes as secondary encoding since slots 3-4 are below 3:1 contrast.
COLORS = {"zero": "#2a78d6", "mean": "#eb6834", "resample": "#1baf7a", "optimal": "#eda100"}
MARKERS = {"zero": "o", "mean": "s", "resample": "D", "optimal": "^"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
THRESHOLD_FRAC = 0.01


def load(dirpath: Path) -> list[dict]:
    runs = [json.loads(p.read_text()) for p in sorted(dirpath.glob("seed*.json"))]
    if not runs:
        sys.exit(f"no seed*.json in {dirpath}")
    return runs


def head_table(run: dict) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    heads = list(run["single"]["zero"].keys())
    circuit = {f"{l}.{h}" for l, h in run["reference_circuit"]}
    labels = np.array([h in circuit for h in heads])
    eff = {m: np.array([run["single"][m][h]["mean"] for h in heads]) for m in METHODS}
    ci = {m: np.array([[run["single"][m][h]["lo"], run["single"][m][h]["hi"]] for h in heads]) for m in METHODS}
    return heads, labels, eff, ci


def head_kind(head: str, in_circuit: bool) -> str:
    layer = int(head.split(".")[0])
    if not in_circuit:
        return "non-circuit"
    return "prev-token (L0)" if layer == 0 else "induction (L1)"


def fmt(xs, digits=2) -> str:
    xs = np.asarray(xs, dtype=float)
    return f"{xs.mean():.{digits}f} ± {xs.std(ddof=1) if len(xs) > 1 else 0:.{digits}f}"


def analyze(runs: list[dict]) -> dict:
    out: dict = {"n_seeds": len(runs), "per_seed": []}
    pair_rho: dict[str, list[float]] = {f"{a}~{b}": [] for a, b in combinations(METHODS, 2)}
    auc: dict[str, list[float]] = {m: [] for m in METHODS}
    prec: dict[str, list[float]] = {m: [] for m in METHODS}
    nc_over_c: dict[str, list[float]] = {m: [] for m in METHODS}
    ratio_to_mean: dict[str, dict[str, list[float]]] = {
        m: {"prev-token (L0)": [], "induction (L1)": []} for m in METHODS if m != "mean"
    }
    zero_fp_singles = []
    # Threshold rule a practitioner might use: a head "matters" if its single-head effect
    # exceeds 1% of the same method's whole-circuit joint effect.
    thr_fp: dict[str, list[int]] = {m: [] for m in METHODS}
    thr_fn: dict[str, list[int]] = {m: [] for m in METHODS}
    for run in runs:
        heads, labels, eff, _ = head_table(run)
        k = int(labels.sum())
        seed_row = {"seed": run["seed"], "circuit": run["reference_circuit"], "clean": run["clean"]}
        for a, b in combinations(METHODS, 2):
            pair_rho[f"{a}~{b}"].append(spearman(eff[a], eff[b]))
        for m in METHODS:
            auc[m].append(auroc(eff[m], labels))
            truth = labels.astype(float)
            prec[m].append(top_k_overlap(eff[m], truth + 1e-9 * np.arange(len(truth)), k))
            g = run["groups"]
            nc_over_c[m].append(g["non_circuit"][m]["mean"] / g["circuit"][m]["mean"])
            tau = THRESHOLD_FRAC * g["circuit"][m]["mean"]
            thr_fp[m].append(int((eff[m][~labels] > tau).sum()))
            thr_fn[m].append(int((eff[m][labels] <= tau).sum()))
        kinds = [head_kind(h, c) for h, c in zip(heads, labels)]
        for m in ratio_to_mean:
            for kind in ratio_to_mean[m]:
                sel = np.array([kk == kind for kk in kinds])
                if sel.any():
                    ratio_to_mean[m][kind].append(float(np.median(eff[m][sel] / np.maximum(eff["mean"][sel], 1e-6))))
        # non-circuit heads that zero-ablation ranks above the weakest circuit head
        zero_fp_singles.append(int((eff["zero"][~labels] > eff["zero"][labels].min()).sum()))
        out["per_seed"].append(seed_row)

    out["spearman_between_methods"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "per_seed": v} for k, v in pair_rho.items()}
    out["auroc_vs_reference"] = {m: {"mean": float(np.mean(v)), "per_seed": v} for m, v in auc.items()}
    out["precision_at_k"] = {m: {"mean": float(np.mean(v)), "per_seed": v} for m, v in prec.items()}
    out["noncircuit_over_circuit_joint"] = {m: {"mean": float(np.mean(v)), "per_seed": v} for m, v in nc_over_c.items()}
    out["median_ratio_to_mean_ablation"] = {
        m: {kind: {"mean": float(np.mean(v)), "per_seed": v} for kind, v in d.items() if v} for m, d in ratio_to_mean.items()
    }
    out["zero_noncircuit_above_weakest_circuit_head"] = zero_fp_singles
    out["threshold_rule"] = {
        "frac_of_circuit_effect": THRESHOLD_FRAC,
        "false_positives": thr_fp,
        "false_negatives": thr_fn,
        "n_circuit": [len(r["reference_circuit"]) for r in runs],
        "n_heads": [len(r["single"]["zero"]) for r in runs],
    }
    out["groups"] = {
        name: {m: [r["groups"][name][m]["mean"] for r in runs if name in r["groups"]] for m in METHODS}
        for name in ("circuit", "non_circuit", "prev_token_heads", "induction_heads", "layer0", "layer1")
    }
    return out


def write_markdown(runs: list[dict], s: dict, path: Path) -> None:
    L = []
    L.append(f"# Q1 summary ({s['n_seeds']} seeds)\n")
    L.append("Effects are increases in induction loss (nats). Chance-level loss is "
             f"{runs[0]['chance_loss']:.2f} nats; clean loss ≈ "
             f"{np.mean([r['clean_eval_loss'] for r in runs]):.4f}.\n")
    L.append("## Models\n")
    L.append("| seed | induction acc | two-back acc | reference circuit (layer.head) |")
    L.append("|---|---|---|---|")
    for r in runs:
        circ = ", ".join(f"{l}.{h}" for l, h in r["reference_circuit"])
        L.append(f"| {r['seed']} | {r['clean']['induction']['acc']:.4f} | {r['clean']['two_back']['acc']:.4f} | {circ} |")
    L.append("\n## Rank agreement between methods (Spearman ρ over 16 single-head effects, mean ± sd over seeds)\n")
    L.append("| pair | ρ |")
    L.append("|---|---|")
    for k, v in s["spearman_between_methods"].items():
        L.append(f"| {k.replace('~', ' vs ')} | {fmt(v['per_seed'])} |")
    L.append("\n## Recovering the reference circuit from single-head effects\n")
    L.append("AUROC = probability a random circuit head has a larger effect than a random non-circuit head. "
             "Precision@k = fraction of the top-k heads (k = circuit size) that are circuit heads.\n")
    L.append("| method | AUROC | precision@k |")
    L.append("|---|---|---|")
    for m in METHODS:
        L.append(f"| {m} | {fmt(s['auroc_vs_reference'][m]['per_seed'])} | {fmt(s['precision_at_k'][m]['per_seed'])} |")
    tr = s["threshold_rule"]
    L.append(f"\n## Threshold rule: a head \"matters\" if its effect > {tr['frac_of_circuit_effect']:.0%} of that method's whole-circuit effect\n")
    L.append(f"Totals over all seeds ({sum(tr['n_circuit'])} circuit heads, "
             f"{sum(tr['n_heads']) - sum(tr['n_circuit'])} non-circuit heads).\n")
    L.append("| method | false positives (non-circuit flagged) | false negatives (circuit missed) |")
    L.append("|---|---|---|")
    for m in METHODS:
        L.append(f"| {m} | {sum(tr['false_positives'][m])} (per seed {tr['false_positives'][m]}) | "
                 f"{sum(tr['false_negatives'][m])} (per seed {tr['false_negatives'][m]}) |")
    L.append("\n## Joint ablation of head groups (mean over seeds; min–max in brackets)\n")
    L.append("| group | " + " | ".join(METHODS) + " |")
    L.append("|---|" + "---|" * len(METHODS))
    for name, d in s["groups"].items():
        cells = []
        for m in METHODS:
            v = np.array(d[m])
            cells.append(f"{v.mean():.3f} [{v.min():.3f}–{v.max():.3f}]" if len(v) else "–")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("\nFalse-positive ratio = effect(all non-circuit heads) / effect(all circuit heads):\n")
    L.append("| method | ratio |")
    L.append("|---|---|")
    for m in METHODS:
        L.append(f"| {m} | {fmt(s['noncircuit_over_circuit_joint'][m]['per_seed'], 3)} |")
    L.append("\n## Effect size relative to mean ablation (median over circuit heads of that kind; mean ± sd over seeds)\n")
    L.append("| method | prev-token heads | induction heads |")
    L.append("|---|---|---|")
    for m, d in s["median_ratio_to_mean_ablation"].items():
        row = [fmt(d[k]["per_seed"], 2) if k in d else "–" for k in ("prev-token (L0)", "induction (L1)")]
        L.append(f"| {m} / mean | " + " | ".join(row) + " |")
    L.append("\n## Per-seed single-head effects\n")
    for r in runs:
        heads, labels, eff, ci = head_table(r)
        L.append(f"<details><summary>seed {r['seed']}</summary>\n")
        L.append("| head | in circuit | prev-token score | induction score | " + " | ".join(METHODS) + " |")
        L.append("|---|---|---|---|" + "---|" * len(METHODS))
        hs = r["head_scores"]
        for i, h in enumerate(heads):
            l, hh = map(int, h.split("."))
            cells = [f"{eff[m][i]:.4f} [{ci[m][i, 0]:.4f}, {ci[m][i, 1]:.4f}]" for m in METHODS]
            L.append(f"| {h} | {'yes' if labels[i] else ''} | {hs['prev_token'][l][hh]:.2f} | {hs['induction'][l][hh]:.2f} | " + " | ".join(cells) + " |")
        L.append("\n</details>\n")
    path.write_text("\n".join(L) + "\n")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def plot_single(runs: list[dict], path: Path) -> None:
    kinds = ["prev-token (L0)", "induction (L1)", "non-circuit"]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200, facecolor=SURFACE)
    _style(ax)
    width = 0.8 / len(METHODS)
    rng = np.random.default_rng(0)
    floor = 1e-4
    for mi, m in enumerate(METHODS):
        for ki, kind in enumerate(kinds):
            ys = []
            for r in runs:
                heads, labels, eff, _ = head_table(r)
                ys += [e for h, c, e in zip(heads, labels, eff[m]) if head_kind(h, c) == kind]
            ys = np.maximum(np.array(ys), floor)
            x = ki + (mi - (len(METHODS) - 1) / 2) * width
            jitter = rng.uniform(-width * 0.3, width * 0.3, len(ys))
            ax.scatter(x + jitter, ys, s=22, color=COLORS[m], marker=MARKERS[m], edgecolor=SURFACE, linewidth=0.8,
                       label=m if ki == 0 else None, zorder=3)
            ax.hlines(np.median(ys), x - width * 0.4, x + width * 0.4, color=INK, linewidth=1.5, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(range(len(kinds)), [f"{k}" for k in kinds], color=INK)
    ax.set_ylabel("loss increase from ablating one head (nats, log)", color=INK2, fontsize=9)
    ax.set_title("Single-head ablation effects by head type, all seeds (bar = median)", color=INK, fontsize=11, loc="left")
    ax.text(0.0, -0.16, f"Values below {floor:g} (including negative) are drawn at {floor:g}.", transform=ax.transAxes,
            fontsize=8, color=INK2)
    leg = ax.legend(title="ablation method", frameon=False, fontsize=9, title_fontsize=9, loc="upper right")
    plt.setp(leg.get_texts(), color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_groups(s: dict, path: Path) -> None:
    groups = [("prev_token_heads", "prev-token heads"), ("induction_heads", "induction heads"),
              ("circuit", "whole circuit"), ("non_circuit", "all non-circuit heads")]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200, facecolor=SURFACE)
    _style(ax)
    width = 0.8 / len(METHODS)
    for mi, m in enumerate(METHODS):
        for gi, (g, _) in enumerate(groups):
            v = np.maximum(np.array(s["groups"][g][m]), 1e-4)
            x = gi + (mi - (len(METHODS) - 1) / 2) * width
            ax.bar(x, v.mean(), width=width - 0.02, color=COLORS[m], label=m if gi == 0 else None, zorder=2)
            ax.scatter(np.full(len(v), x), v, s=10, color=INK, zorder=3, linewidth=0)
    ax.set_yscale("log")
    ax.set_xticks(range(len(groups)), [lbl for _, lbl in groups], color=INK)
    ax.set_ylabel("loss increase from joint ablation (nats, log)", color=INK2, fontsize=9)
    ax.set_title("Joint ablation of head groups (bar = mean, dots = seeds)", color=INK, fontsize=11, loc="left")
    leg = ax.legend(title="ablation method", frameon=False, fontsize=9, title_fontsize=9, loc="upper left", ncols=4)
    plt.setp(leg.get_texts(), color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/q1")
    runs = load(d)
    s = analyze(runs)
    (d / "summary.json").write_text(json.dumps(s, indent=1))
    write_markdown(runs, s, d / "summary.md")
    plot_single(runs, d / "single_head_effects.png")
    plot_groups(s, d / "group_effects.png")
    print((d / "summary.md").read_text().split("## Per-seed")[0])


if __name__ == "__main__":
    main()
