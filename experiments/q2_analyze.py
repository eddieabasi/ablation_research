"""Aggregate Q2 results across seeds into summary.md / summary.json and figures.

    python experiments/q2_analyze.py results/q2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ablation.stats import spearman

METHODS = ("zero", "mean", "shift")
# Reference palette slots 1-3 (validated all-pairs, light mode). Q2 has no resample, so
# slot 3 is used for shift; marker shapes differ from Q1's as a second cue.
COLORS = {"zero": "#2a78d6", "mean": "#eb6834", "shift": "#1baf7a"}
MARKERS = {"zero": "o", "mean": "s", "shift": "P"}
LABELS = {"zero": "zero (remove both)", "mean": "mean (remove information)", "shift": "shift (remove constant)"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GROUPS = {
    "group:non_circuit": "non-circuit heads",
    "group:prev_token_heads": "prev-token heads",
    "group:induction_heads": "induction heads",
    "group:layer0": "all of layer 0",
}


def load(d: Path) -> list[dict]:
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("seed*.json"))]
    if not runs:
        sys.exit(f"no seed*.json in {d}")
    return runs


def kind(run: dict, unit: str) -> str | None:
    if unit.startswith("group:"):
        return None
    circ = {f"{l}.{h}" for l, h in run["reference_circuit"]}
    if unit not in circ:
        return "non-circuit"
    return "prev-token" if unit.startswith("0.") else "induction"


def eff(u: dict, key: str) -> float:
    return u[key]["effect"]


def decomposition(u: dict) -> dict:
    z, m, s = eff(u, "zero"), eff(u, "mean"), eff(u, "shift")
    inter = z - m - s
    # Shapley split of the zero effect between "constant" and "information"
    const = 0.5 * (s + (z - m))
    info = 0.5 * (m + (z - s))
    return {"zero": z, "mean": m, "shift": s, "interaction": inter,
            "shapley_constant": const, "shapley_information": info,
            "constant_share": const / z if z > 1e-4 else float("nan")}


def summarize(runs: list[dict]) -> dict:
    out: dict = {"n_seeds": len(runs)}

    # 1. decomposition: groups (per seed) and single heads by kind (pooled)
    dec_groups = {g: [decomposition(r["units"][g]) for r in runs if g in r["units"]] for g in GROUPS}
    out["decomposition_groups"] = {
        g: {k: [d[k] for d in ds] for k in ds[0]} for g, ds in dec_groups.items() if ds
    }
    dec_single: dict[str, list[dict]] = {"prev-token": [], "induction": [], "non-circuit": []}
    for r in runs:
        for unit, u in r["units"].items():
            k = kind(r, unit)
            if k:
                dec_single[k].append(decomposition(u))
    out["decomposition_single"] = {k: {kk: [d[kk] for d in ds] for kk in ds[0]} for k, ds in dec_single.items()}

    # 2. LayerNorm: frozen / live per method
    ln: dict = {}
    for g in GROUPS:
        ln[g] = {m: [eff(r["units"][g], f"{m}_frozen_ln") / eff(r["units"][g], m) if eff(r["units"][g], m) > 1e-3 else float("nan")
                     for r in runs if g in r["units"]]
                 for m in METHODS}
    ln_single: dict = {k: {m: [] for m in METHODS} for k in dec_single}
    for r in runs:
        for unit, u in r["units"].items():
            k = kind(r, unit)
            if k:
                for m in METHODS:
                    live = eff(u, m)
                    if live > 1e-3:  # ratios of near-zero effects are noise
                        ln_single[k][m].append(eff(u, f"{m}_frozen_ln") / live)
    out["ln_frozen_over_live_groups"] = ln
    out["ln_frozen_over_live_single"] = ln_single
    out["ln_frozen_gt_live_fraction"] = {
        m: float(np.mean([eff(u, f"{m}_frozen_ln") > eff(u, m) for r in runs for unit, u in r["units"].items()
                          if not unit.startswith("group:") and eff(u, m) > 1e-3]))
        for m in METHODS
    }

    # 3. paths (layer-0 units): effect via each receiver as a fraction of the full effect
    paths: dict = {}
    for g in ("group:non_circuit", "group:prev_token_heads", "group:layer0"):
        paths[g] = {}
        for m in METHODS:
            rows = []
            for r in runs:
                u = r["units"].get(g)
                if u and f"{m}_paths" in u:
                    full = eff(u, m)
                    rows.append({rc: v / full for rc, v in u[f"{m}_paths"].items()} | {"full": full})
            if rows:
                paths[g][m] = {k: [row[k] for row in rows] for k in rows[0]}
    out["paths_fraction_of_full"] = paths

    # 4. diagnostics for groups
    diag_keys = ("induction_attention", "output_entropy", "accuracy", "ln_q_scale_ratio", "ln_final_scale_ratio")
    out["clean_diagnostics"] = {k: [r["clean"][k] for r in runs] for k in ("induction_attention", "output_entropy", "accuracy")}
    out["diagnostics_groups"] = {
        g: {f"{m}{suffix}": {k: [r["units"][g][f"{m}{suffix}"][k] for r in runs if g in r["units"]] for k in diag_keys}
            for m in METHODS for suffix in ("", "_frozen_ln")}
        for g in GROUPS
    }

    # 5. predictors of the constant's damage, layer-0 single heads pooled
    pred: dict = {}
    for subset in ("non-circuit", "all-layer0"):
        xs: dict[str, list[float]] = {k: [] for k in ("mean_norm", "q_read", "k_read", "v_read")}
        ys: dict[str, list[float]] = {"shift": [], "zero_minus_mean": []}
        for r in runs:
            sr = r["static_reads_layer0"]
            for h in range(len(sr["mean_norm"])):
                unit = f"0.{h}"
                if subset == "non-circuit" and kind(r, unit) != "non-circuit":
                    continue
                u = r["units"][unit]
                for k in xs:
                    xs[k].append(sr[k][h])
                ys["shift"].append(eff(u, "shift"))
                ys["zero_minus_mean"].append(eff(u, "zero") - eff(u, "mean"))
        pred[subset] = {"n": len(ys["shift"]),
                        **{f"{y}~{x}": spearman(xs[x], ys[y]) for y in ys for x in xs}}
    out["predictors"] = pred
    return out


def fmt(xs, d=2) -> str:
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    if len(xs) == 0:
        return "–"
    return f"{xs.mean():.{d}f} ± {xs.std(ddof=1):.{d}f}" if len(xs) > 1 else f"{xs.mean():.{d}f}"


def med(xs, d=3, show_n=False) -> str:
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    if len(xs) == 0:
        return "–"
    lo, hi = np.quantile(xs, [0.25, 0.75])
    n = f" (n={len(xs)})" if show_n else ""
    return f"{np.median(xs):.{d}f} [{lo:.{d}f}, {hi:.{d}f}]{n}"


def write_markdown(runs: list[dict], s: dict, path: Path) -> None:
    L: list[str] = [f"# Q2 summary ({s['n_seeds']} seeds)\n"]
    L.append("Effects are increases in induction loss (nats). `mean` removes a head's input-dependent "
             "information, `shift` removes its constant (mean) output, `zero` removes both. "
             "Interaction = zero − mean − shift. Shapley share of the constant = "
             "½[shift + (zero − mean)] / zero.\n")
    L.append("## 1. Decomposing zero ablation\n")
    L.append("Groups (mean ± sd over seeds):\n")
    L.append("| group | zero | mean (info) | shift (constant) | interaction | constant's Shapley share |")
    L.append("|---|---|---|---|---|---|")
    for g, lbl in GROUPS.items():
        d = s["decomposition_groups"].get(g)
        if d:
            L.append(f"| {lbl} | {fmt(d['zero'])} | {fmt(d['mean'], 3)} | {fmt(d['shift'], 3)} | {fmt(d['interaction'])} | {fmt(d['constant_share'])} |")
    L.append("\nSingle heads (median [IQR], pooled over seeds):\n")
    L.append("| head type | n | zero | mean | shift | interaction | constant's Shapley share |")
    L.append("|---|---|---|---|---|---|---|")
    for k, d in s["decomposition_single"].items():
        L.append(f"| {k} | {len(d['zero'])} | {med(d['zero'], 4)} | {med(d['mean'], 4)} | {med(d['shift'], 4)} | {med(d['interaction'], 4)} | {med(d['constant_share'], 2)} |")

    L.append("\n## 2. Does LayerNorm mediate it? (effect with LayerNorm frozen at clean scale ÷ live effect)\n")
    L.append("Ratio > 1: freezing LayerNorm makes the ablation *worse*, i.e. live LayerNorm was cushioning it. "
             "Only ablations with a live effect above 0.001 nats are included (ratios of near-zero effects are noise).\n")
    L.append("| unit | zero | mean | shift |")
    L.append("|---|---|---|---|")
    for g, lbl in GROUPS.items():
        d = s["ln_frozen_over_live_groups"][g]
        L.append(f"| {lbl} (group) | {fmt(d['zero'])} | {fmt(d['mean'])} | {fmt(d['shift'])} |")
    for k, d in s["ln_frozen_over_live_single"].items():
        L.append(f"| {k} (single, median [IQR]) | {med(d['zero'], 2, True)} | {med(d['mean'], 2, True)} | {med(d['shift'], 2, True)} |")
    fr = s["ln_frozen_gt_live_fraction"]
    L.append(f"\nFraction of single-head ablations (effect > 0.001) made worse by freezing LayerNorm: "
             f"zero {fr['zero']:.0%}, mean {fr['mean']:.0%}, shift {fr['shift']:.0%}.\n")

    L.append("## 3. Which downstream reads carry the damage? (layer-0 groups; path effect ÷ full effect)\n")
    L.append("Each column delivers the ablation to one receiver only: layer-1 queries (q), keys (k), "
             "values (v), the unembedding directly (direct), or queries and keys together (q+k).\n")
    L.append("| unit | method | full effect | q | k | v | direct | q+k |")
    L.append("|---|---|---|---|---|---|---|---|")
    for g, dm in s["paths_fraction_of_full"].items():
        for m, d in dm.items():
            L.append(f"| {GROUPS[g]} | {m} | {fmt(d['full'])} | {fmt(d['q'])} | {fmt(d['k'])} | {fmt(d['v'])} | {fmt(d['direct'])} | {fmt(d['qk'])} |")

    L.append("\n## 4. What the induction heads do under each ablation (groups, mean ± sd over seeds)\n")
    c = s["clean_diagnostics"]
    L.append(f"Clean: induction attention {fmt(c['induction_attention'])}, output entropy "
             f"{fmt(c['output_entropy'], 3)} nats, accuracy {fmt(c['accuracy'], 4)}.\n")
    L.append("Induction attention = layer-1 attention mass on the key holding the answer, averaged over "
             "heads and induction positions. LN ratios = LayerNorm scale relative to clean (1 by "
             "construction when frozen).\n")
    L.append("| unit | condition | induction attention | output entropy | accuracy | layer-1 LN scale | final LN scale |")
    L.append("|---|---|---|---|---|---|---|")
    for g, lbl in GROUPS.items():
        for cond, d in s["diagnostics_groups"][g].items():
            if not d["accuracy"]:
                continue
            L.append(f"| {lbl} | {cond.replace('_frozen_ln', ', LN frozen')} | {fmt(d['induction_attention'])} | "
                     f"{fmt(d['output_entropy'])} | {fmt(d['accuracy'])} | {fmt(d['ln_q_scale_ratio'])} | {fmt(d['ln_final_scale_ratio'])} |")

    L.append("\n## 5. What predicts the damage from removing a layer-0 head's constant? (Spearman ρ, pooled)\n")
    L.append("`q_read` / `k_read` / `v_read`: how far the head's mean output moves the layer-1 "
             "queries / keys / values (first-order, through the clean LayerNorm scale). "
             "`mean_norm`: size of the mean output itself.\n")
    L.append("| heads | n | target | mean_norm | q_read | k_read | v_read |")
    L.append("|---|---|---|---|---|---|---|")
    for subset, d in s["predictors"].items():
        for y in ("shift", "zero_minus_mean"):
            L.append(f"| {subset} | {d['n']} | {y} | " + " | ".join(f"{d[f'{y}~{x}']:.2f}" for x in ("mean_norm", "q_read", "k_read", "v_read")) + " |")
    path.write_text("\n".join(L) + "\n")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def plot_decomposition(s: dict, path: Path) -> None:
    groups = ["group:non_circuit", "group:prev_token_heads", "group:layer0"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), dpi=200, facecolor=SURFACE)
    order = ["mean", "shift", "zero"]
    for ax, g in zip(axes, groups):
        _style(ax)
        d = s["decomposition_groups"][g]
        for i, m in enumerate(order):
            v = np.array(d[m])
            ax.bar(i, v.mean(), width=0.6, color=COLORS[m], zorder=2)
            ax.scatter(np.full(len(v), i), v, s=9, color=INK, zorder=3, linewidth=0)
        add = np.array(d["mean"]) + np.array(d["shift"])
        ax.hlines(add.mean(), 2 - 0.36, 2 + 0.36, color=INK, linewidth=1.5, zorder=4)
        ax.annotate("mean + shift", (2 + 0.38, add.mean()), fontsize=7, color=INK2, va="center")
        ax.set_xticks(range(3), ["remove\ninformation\n(mean)", "remove\nconstant\n(shift)", "remove\nboth\n(zero)"], fontsize=8, color=INK)
        ax.set_title(GROUPS[g], fontsize=10, color=INK, loc="left")
        ax.set_xlim(-0.6, 2.9)
    axes[0].set_ylabel("loss increase (nats)", color=INK2, fontsize=9)
    fig.suptitle("Zero ablation is not 'information + mean shift': the two interact (bars = mean over seeds, dots = seeds)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_layernorm(runs: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), dpi=200, facecolor=SURFACE, sharex=True, sharey=True)
    lo, hi = 1e-3, 1e-3
    for r in runs:
        for unit, u in r["units"].items():
            if not unit.startswith("group:"):
                hi = max(hi, *(eff(u, k) for m in METHODS for k in (m, f"{m}_frozen_ln")))
    for ax, m in zip(axes, METHODS):
        _style(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=1)
        live, frozen = [], []
        for r in runs:
            for unit, u in r["units"].items():
                if not unit.startswith("group:") and eff(u, m) > lo:
                    live.append(eff(u, m))
                    frozen.append(max(eff(u, f"{m}_frozen_ln"), lo))
        ax.plot([lo, hi * 2], [lo, hi * 2], color=INK2, linewidth=1, zorder=1)
        ax.scatter(live, frozen, s=16, color=COLORS[m], marker=MARKERS[m], edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(LABELS[m], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("effect, LayerNorm live (nats)", fontsize=8, color=INK2)
        above = np.mean(np.array(frozen) > np.array(live)) if live else float("nan")
        ax.text(0.03, 0.93, f"{above:.0%} of {len(live)} above the diagonal", transform=ax.transAxes, fontsize=7.5, color=INK2)
    axes[0].set_ylabel("effect, LayerNorm frozen (nats)", fontsize=8, color=INK2)
    fig.suptitle("Freezing LayerNorm mostly makes single-head ablations worse: LayerNorm cushions them, it doesn't cause them",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, f"Single heads with a live effect above {lo:g} nats, all seeds. Frozen values below {lo:g} drawn at {lo:g}. Line = no change.",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_paths(s: dict, path: Path) -> None:
    conds = [("group:non_circuit", "zero"), ("group:non_circuit", "shift"),
             ("group:prev_token_heads", "mean"), ("group:prev_token_heads", "zero")]
    recv = ["q", "k", "v", "direct", "qk"]
    rlab = ["q", "k", "v", "direct", "q+k"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.4), dpi=200, facecolor=SURFACE, sharey=True)
    for ax, (g, m) in zip(axes, conds):
        _style(ax)
        d = s["paths_fraction_of_full"][g][m]
        for i, rc in enumerate(recv):
            v = np.array(d[rc])
            ax.bar(i, v.mean(), width=0.6, color=COLORS[m], zorder=2)
            ax.scatter(np.full(len(v), i), v, s=8, color=INK, zorder=3, linewidth=0)
        ax.axhline(1.0, color=INK2, linewidth=1, zorder=1)
        ax.set_xticks(range(len(recv)), rlab, fontsize=8, color=INK)
        ax.set_title(f"{GROUPS[g]}\n{m} ablation", fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("path effect ÷ full effect", fontsize=8, color=INK2)
    fig.suptitle("Where the damage travels: layer-1 queries for the constant, keys for previous-token information",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Line at 1 = the whole effect. Bars = mean over seeds, dots = seeds.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/q2")
    runs = load(d)
    s = summarize(runs)
    (d / "summary.json").write_text(json.dumps(s, indent=1))
    write_markdown(runs, s, d / "summary.md")
    plot_decomposition(s, d / "decomposition.png")
    plot_layernorm(runs, d / "layernorm.png")
    plot_paths(s, d / "paths.png")
    print((d / "summary.md").read_text())


if __name__ == "__main__":
    main()
