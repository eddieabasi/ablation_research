"""Aggregate Q4 results (results/q4/h<H>_seed<k>.json) into summary.md / summary.json and figures.

    python experiments/q4_analyze.py results/q4
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Reference palette slots 1-3 (validated all-pairs, light mode), one per head count.
H_COLORS = {2: "#2a78d6", 4: "#eb6834", 8: "#1baf7a"}
H_MARKERS = {2: "o", 4: "s", 8: "D"}
KINDS = ["ind-ind", "prev-prev", "prev-ind", "circuit-non", "non-non"]
KIND_LABELS = {
    "ind-ind": "induction +\ninduction",
    "prev-prev": "prev-token +\nprev-token",
    "prev-ind": "prev-token +\ninduction",
    "circuit-non": "circuit +\nnon-circuit",
    "non-non": "non-circuit +\nnon-circuit",
}
THRESHOLD = 0.01


def usable(r: dict) -> tuple[bool, str]:
    """A model is analysed only if it learned both tasks and has a complete reference circuit."""
    if not r.get("training", {}).get("met_criterion", True):
        return False, f"missed 99% accuracy after {r['training'].get('steps')} steps"
    if not heads_of(r, "prev") or not heads_of(r, "ind"):
        return False, "no complete reference circuit"
    return True, ""


def load(d: Path) -> tuple[dict[int, list[dict]], list[dict]]:
    runs: dict[int, list[dict]] = defaultdict(list)
    excluded = []
    for p in sorted(d.glob("h*_seed*.json")):
        r = json.loads(p.read_text())
        ok, why = usable(r)
        if ok:
            runs[r["heads"]].append(r)
        else:
            excluded.append({"heads": r["heads"], "seed": r["seed"], "reason": why,
                             "induction_acc": r["clean"]["induction"]["acc"]})
    if not runs:
        sys.exit(f"no usable h*_seed*.json in {d}")
    return dict(sorted(runs.items())), excluded


def fmt(xs, d=2) -> str:
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    if len(xs) == 0:
        return "–"
    return f"{xs.mean():.{d}f} ± {xs.std(ddof=1):.{d}f}" if len(xs) > 1 else f"{xs.mean():.{d}f}"


def med(xs, d=3) -> str:
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    if len(xs) == 0:
        return "–"
    return f"{np.median(xs):.{d}f}"


def heads_of(r: dict, kind: str) -> list[str]:
    return [f"{l}.{h}" for l, h in r["reference_circuit"] if (l == 0) == (kind == "prev")]


def hidden_backups(rs: list[dict]) -> list[dict]:
    """For every non-circuit layer-0 head: its single effect, and its effect once its strongest
    reference previous-token partner P is ablated (effect(head and P) - effect(P)), both relative to
    the whole-circuit effect. A head is *hidden* if it misses the 1% threshold alone but passes it
    once P is gone (and the interaction's CI excludes 0)."""
    rows = []
    for r in rs:
        total = r["groups"]["circuit"]["mean"]["effect"]
        prev = {tuple(map(int, h.split("."))) for h in heads_of(r, "prev")}
        circ = {tuple(h) for h in r["reference_circuit"]}
        by_head = defaultdict(list)
        for p in r["pairs"]:
            a, b = tuple(p["a"]), tuple(p["b"])
            for x, y in ((a, b), (b, a)):
                if x[0] == 0 and x not in circ and y in prev:
                    by_head[x].append((y, p["mean"]))
        for (l, h), partners in sorted(by_head.items()):
            y, m = max(partners, key=lambda t: t[1]["interaction"])
            single = r["single"][f"{l}.{h}"]["mean"]["effect"]
            partner_single = r["single"][f"{y[0]}.{y[1]}"]["mean"]["effect"]
            rows.append({
                "seed": r["seed"], "head": f"{l}.{h}", "partner": f"{y[0]}.{y[1]}",
                "prev_token_score": r["head_scores"]["prev_token"][l][h],
                "bos_score": r["head_scores"]["bos"][l][h],
                "single_rel": single / total,
                "after_partner_rel": (m["effect_AB"] - partner_single) / total,
                "interaction": m["interaction"], "interaction_lo": m["interaction_lo"],
                "margin_interaction": m["margin_test"]["margin_interaction"],
                "hidden": bool(single < THRESHOLD * total and m["effect_AB"] - partner_single > THRESHOLD * total
                               and m["interaction_lo"] > 0),
            })
    return rows


def _spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def summarize(runs: dict[int, list[dict]]) -> dict:
    out: dict = {}
    for H, rs in runs.items():
        o: dict = {"n_seeds": len(rs), "seeds": [r["seed"] for r in rs]}
        o["clean_induction_acc"] = [r["clean"]["induction"]["acc"] for r in rs]
        o["clean_two_back_acc"] = [r["clean"]["two_back"]["acc"] for r in rs]
        o["train_steps"] = [r.get("training", {}).get("steps") for r in rs]
        o["n_prev"] = [len(heads_of(r, "prev")) for r in rs]
        o["n_ind"] = [len(heads_of(r, "ind")) for r in rs]

        # detection by single-head ablation (mean), threshold = 1% of whole-circuit effect
        for method in ("mean", "resample"):
            rec = {"prev": [], "ind": []}
            frac_single = {"prev": [], "ind": []}
            sum_over_joint = {"prev": [], "ind": []}
            second_order = {"prev": [], "ind": []}
            for r in rs:
                total = r["groups"]["circuit"][method]["effect"]
                for kind, gname in (("prev", "prev_token_heads"), ("ind", "induction_heads")):
                    hs = heads_of(r, kind)
                    if not hs:
                        continue
                    effs = np.array([r["single"][h][method]["effect"] for h in hs])
                    rec[kind].append(float(np.mean(effs > THRESHOLD * total)))
                    joint = r["groups"][gname][method]["effect"]
                    frac_single[kind] += list(effs / joint)
                    sum_over_joint[kind].append(float(effs.sum() / joint))
                    # second-order inclusion-exclusion: singles + every pairwise interaction within the kind
                    kk = "prev-prev" if kind == "prev" else "ind-ind"
                    inter = sum(p[method]["interaction"] for p in r["pairs"] if p["kind"] == kk)
                    second_order[kind].append(float((effs.sum() + inter) / joint))
            o[f"{method}_recall"] = rec
            o[f"{method}_single_over_group"] = frac_single
            o[f"{method}_sum_singles_over_group"] = sum_over_joint
            o[f"{method}_second_order_over_group"] = second_order

        # pair interactions by kind
        pk: dict = {}
        for method in ("mean", "resample"):
            pk[method] = {}
            for kind in KINDS:
                rows = [p[method] for r in rs for p in r["pairs"] if p["kind"] == kind]
                if not rows:
                    continue
                inter = np.array([x["interaction"] for x in rows])
                lo = np.array([x["interaction_lo"] for x in rows])
                hi = np.array([x["interaction_hi"] for x in rows])
                ab = np.array([x["effect_AB"] for x in rows])
                mt = [x["margin_test"] for x in rows]
                pk[method][kind] = {
                    "n": len(rows),
                    "interaction": inter.tolist(),
                    "relative": (inter / np.maximum(ab, 1e-6)).tolist(),
                    "sig_pos": float(np.mean(lo > 0)),
                    "sig_neg": float(np.mean(hi < 0)),
                    "margin_neg": float(np.mean([m["margin_interaction"] < 0 for m in mt])),
                    "margin_interaction": [m["margin_interaction"] for m in mt],
                    "explained": [m["explained"] for m in mt],
                    "effect_AB": ab.tolist(),
                }
        o["pairs"] = pk

        # nested curves (induction heads), normalised by the all-induction effect
        curves = []
        for r in rs:
            c = r["nested"].get("induction_heads")
            if not c:
                continue
            n = len(c)
            full = np.mean(c[-1]["effects"])
            curves.append({"x": [p["k"] / n for p in c], "y": [float(np.mean(p["effects"]) / full) for p in c],
                           "raw": [float(np.mean(p["effects"])) for p in c], "n": n})
        o["nested_induction"] = curves
        # how many induction heads (fraction) can go before effect exceeds 10% of full
        o["removable_before_10pct"] = [
            max([0.0] + [x for x, y in zip(cv["x"], cv["y"]) if y <= 0.10]) for cv in curves
        ]

        # sufficiency: keep one, ablate the rest of that kind (mean), relative to ablating all of that kind
        suff = {"ind": [], "prev": []}
        for r in rs:
            for kind, gname in (("ind", "induction_heads"), ("prev", "prev_token_heads")):
                s = r["sufficiency"].get(gname)
                if s:
                    full = r["groups"][gname]["mean"]["effect"]
                    suff[kind] += [v / full for v in s.values()]
        o["sufficiency_keep_one_over_none"] = suff

        # hidden backups: non-circuit layer-0 heads vs the reference previous-token heads (mean ablation)
        o["hidden"] = hidden_backups(rs)
        out[H] = o
    return out


def write_markdown(runs, s, path: Path, excluded=()) -> None:
    L = [f"# Q4 summary\n",
         "Effects are increases in induction loss (nats) on the Q1 evaluation sets. Primary method: mean "
         "ablation. Interaction of a pair = effect(both) − effect(A) − effect(B); significant = 95% bootstrap CI "
         "over sequences excludes 0. Margin interaction: same, on the target-logit margin (> 0 = superadditive "
         "inside the network). Explained = share of the loss interaction that loss curvature alone would "
         "produce if margin drops added.\n"]
    L.append("## Models\n")
    L.append("| heads/layer | seeds | training steps | induction acc | two-back acc | prev-token heads | induction heads |")
    L.append("|---|---|---|---|---|---|---|")
    for H, o in s.items():
        L.append(f"| {H} | {o['n_seeds']} ({', '.join(map(str, o['seeds']))}) | {', '.join(map(str, o['train_steps']))} | "
                 f"{fmt(o['clean_induction_acc'], 4)} | {fmt(o['clean_two_back_acc'], 4)} | "
                 f"{fmt(o['n_prev'], 1)} | {fmt(o['n_ind'], 1)} |")
    if excluded:
        L.append("\nExcluded models:\n")
        for e in excluded:
            L.append(f"* {e['heads']} heads/layer, seed {e['seed']}: {e['reason']} (induction accuracy {e['induction_acc']:.3f})")
    L.append("\n## 1. Does single-head ablation find the circuit heads?\n")
    L.append(f"Recall = fraction of circuit heads whose single-head effect exceeds {THRESHOLD:.0%} of the "
             "whole-circuit effect. Single ÷ group = one head's effect ÷ effect of ablating all heads of its kind "
             "(median over heads). Σ singles ÷ group = 1 if effects add up, ≪ 1 under redundancy. + pairs = the same "
             "with every pairwise interaction within the group added (second-order inclusion–exclusion).\n")
    L.append("| heads/layer | method | recall: prev-token | recall: induction | single ÷ group: induction | Σ singles ÷ group: induction | + pairs ÷ group: induction | Σ singles ÷ group: prev-token |")
    L.append("|---|---|---|---|---|---|---|---|")
    for H, o in s.items():
        for m in ("mean", "resample"):
            L.append(f"| {H} | {m} | {fmt(o[f'{m}_recall']['prev'])} | {fmt(o[f'{m}_recall']['ind'])} | "
                     f"{med(o[f'{m}_single_over_group']['ind'], 3)} | {fmt(o[f'{m}_sum_singles_over_group']['ind'], 3)} | "
                     f"{fmt(o[f'{m}_second_order_over_group']['ind'], 3)} | "
                     f"{fmt(o[f'{m}_sum_singles_over_group']['prev'], 3)} |")
    L.append("\n## 2. Pair interactions by pair type\n")
    L.append("| heads/layer | method | pair type | n | median interaction | median relative (÷ effect of both) | sig. > 0 | sig. < 0 | median margin interaction | margin < 0 | median explained by curvature |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for H, o in s.items():
        for m in ("mean", "resample"):
            for kind in KINDS:
                d = o["pairs"][m].get(kind)
                if d:
                    L.append(f"| {H} | {m} | {kind} | {d['n']} | {med(d['interaction'], 4)} | {med(d['relative'], 2)} | "
                             f"{d['sig_pos']:.0%} | {d['sig_neg']:.0%} | {med(d['margin_interaction'], 3)} | {d['margin_neg']:.0%} | "
                             f"{med(d['explained'], 2)} |")
    L.append("\n## 3. Nested ablation of induction heads (mean ablation)\n")
    L.append("Effect of removing k of n induction heads (mean over random subsets) ÷ effect of removing all n.\n")
    for H, o in s.items():
        if not o["nested_induction"]:
            continue
        L.append(f"**{H} heads/layer** (removable before the effect passes 10% of full: "
                 f"{fmt(o['removable_before_10pct'])} of the induction heads)\n")
        L.append("| seed | n | " + " | ".join(f"k={k}" for k in range(1, max(c["n"] for c in o["nested_induction"]) + 1)) + " |")
        L.append("|---|---|" + "---|" * max(c["n"] for c in o["nested_induction"]))
        for r, cv in zip(runs[H], o["nested_induction"]):
            L.append(f"| {r['seed']} | {cv['n']} | " + " | ".join(f"{y:.3f}" for y in cv["y"]) + " |")
        L.append("")
    L.append("## 4. Sufficiency: keep one head of a kind, ablate the others\n")
    L.append("Effect with one head left ÷ effect with none left (0 = one head is enough, 1 = one head does nothing).\n")
    L.append("| heads/layer | induction heads | prev-token heads |")
    L.append("|---|---|---|")
    for H, o in s.items():
        sf = o["sufficiency_keep_one_over_none"]
        L.append(f"| {H} | {fmt(sf['ind'])} (n={len(sf['ind'])}) | {fmt(sf['prev'])} (n={len(sf['prev'])}) |")
    L.append("\n## 5. Hidden backups outside the reference circuit\n")
    L.append("Non-circuit layer-0 heads (previous-token score below the 0.3 threshold). *Alone* = single-head effect; "
             "*after P* = effect once the head's strongest reference previous-token partner P is already ablated "
             "(effect(head and P) − effect(P)); both ÷ whole-circuit effect. Hidden = alone < 1%, after P > 1%, and the "
             "pair interaction's CI excludes 0. ρ = Spearman between previous-token score and the interaction with P "
             "(pooled over seeds; per-model median in brackets).\n")
    L.append("| heads/layer | non-circuit layer-0 heads | pass 1% alone | pass 1% after P | hidden | hidden, score ≥ 0.1 | hidden, score < 0.1 | median after P ÷ alone | ρ(prev-token score, interaction) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for H, o in s.items():
        hb = o["hidden"]
        if not hb:
            continue
        hid = [x for x in hb if x["hidden"]]
        ratio = [x["after_partner_rel"] / max(x["single_rel"], 1e-6) for x in hb]
        per_model = [_spearman([x["prev_token_score"] for x in hb if x["seed"] == sd],
                               [x["interaction"] for x in hb if x["seed"] == sd]) for sd in o["seeds"]]
        rho = _spearman([x["prev_token_score"] for x in hb], [x["interaction"] for x in hb])
        hi_s = [x for x in hb if x["prev_token_score"] >= 0.1]
        lo_s = [x for x in hb if x["prev_token_score"] < 0.1]
        L.append(f"| {H} | {len(hb)} | {sum(x['single_rel'] > THRESHOLD for x in hb)} | "
                 f"{sum(x['after_partner_rel'] > THRESHOLD for x in hb)} | {len(hid)} | "
                 f"{sum(x['hidden'] for x in hi_s)}/{len(hi_s)} | {sum(x['hidden'] for x in lo_s)}/{len(lo_s)} | {med(ratio, 0)} | "
                 f"{rho:.2f} ({med(per_model, 2)}) |")
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


def plot_nested(s: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    for H, o in s.items():
        for i, cv in enumerate(o["nested_induction"]):
            x = [0.0] + cv["x"]
            y = [0.0] + cv["y"]
            ax.plot(x, y, color=H_COLORS[H], linewidth=1.2, alpha=0.45, zorder=2)
        xs = sorted({x for cv in o["nested_induction"] for x in cv["x"]})
        if xs:
            mean_y = [np.mean([y for cv in o["nested_induction"] for xx, y in zip(cv["x"], cv["y"]) if abs(xx - x) < 1e-9]) for x in xs]
            ax.plot([0.0] + xs, [0.0] + mean_y, color=H_COLORS[H], linewidth=2.2, marker=H_MARKERS[H], markersize=5,
                    markeredgecolor=SURFACE, label=f"{H} heads/layer", zorder=3)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, zorder=1)
    ax.text(0.52, 0.47, "additive", rotation=33, fontsize=7.5, color=INK2)
    ax.set_xlabel("fraction of induction heads ablated", fontsize=9, color=INK2)
    ax.set_ylabel("effect ÷ effect of ablating all of them", fontsize=9, color=INK2)
    ax.set_title("Removing induction heads: little happens until most are gone", fontsize=10, color=INK, loc="left")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    plt.setp(leg.get_texts(), color=INK)
    fig.text(0.01, 0.005, "Mean ablation; thin lines = seeds, thick = mean.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_pairs(s: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200, facecolor=SURFACE)
    rng = np.random.default_rng(0)
    for ax, key, ylab in ((axes[0], "relative", "loss interaction ÷ effect of ablating both"),
                          (axes[1], "margin_interaction", "margin interaction (nats)")):
        _style(ax)
        width = 0.8 / len(s)
        for hi, (H, o) in enumerate(s.items()):
            for ki, kind in enumerate(KINDS):
                d = o["pairs"]["mean"].get(kind)
                if not d:
                    continue
                y = np.array(d[key])
                x = ki + (hi - (len(s) - 1) / 2) * width + rng.uniform(-width * 0.3, width * 0.3, len(y))
                ax.scatter(x, y, s=10, color=H_COLORS[H], marker=H_MARKERS[H], edgecolor=SURFACE, linewidth=0.4,
                           label=f"{H} heads/layer" if ki == 0 else None, zorder=3)
        ax.axhline(0, color=INK2, linewidth=1, zorder=1)
        ax.set_xticks(range(len(KINDS)), [KIND_LABELS[k] for k in KINDS], fontsize=7.5, color=INK)
        ax.set_ylabel(ylab, fontsize=8, color=INK2)
        if key == "margin_interaction":
            ax.set_yscale("symlog", linthresh=0.05)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_title("Loss: > 0 superadditive (backups), < 0 subadditive", fontsize=9, color=INK, loc="left")
    axes[1].set_title("Inside the network (target-logit margin)", fontsize=9, color=INK, loc="left")
    leg = axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    plt.setp(leg.get_texts(), color=INK)
    fig.suptitle("Pairs of heads that back each other up interact superadditively; heads in series do not",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Every pair of heads, mean ablation, all seeds. Right panel: symmetric log scale, linear within ±0.05.",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_hidden(s: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200, facecolor=SURFACE)
    _style(ax)
    floor = 1e-5
    for H, o in s.items():
        hb = o["hidden"]
        if not hb:
            continue
        x = np.array([r["prev_token_score"] for r in hb])
        a = np.maximum([r["single_rel"] for r in hb], floor)
        b = np.maximum([r["after_partner_rel"] for r in hb], floor)
        ax.vlines(x, a, b, color=H_COLORS[H], linewidth=0.8, alpha=0.5, zorder=2)
        ax.scatter(x, a, s=16, facecolor=SURFACE, edgecolor=H_COLORS[H], marker=H_MARKERS[H], linewidth=1, zorder=3)
        ax.scatter(x, b, s=16, color=H_COLORS[H], marker=H_MARKERS[H], edgecolor=SURFACE, linewidth=0.4, zorder=3,
                   label=f"{H} heads/layer")
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.8, 1.5)
    ax.axhline(THRESHOLD, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.186, THRESHOLD * 0.72, "1% of circuit effect", fontsize=7, color=INK2, va="top")
    ax.axvline(0.3, color=INK2, linewidth=1, zorder=1)
    ax.text(0.296, floor * 1.3, "reference-circuit\nthreshold", fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_xlim(-0.01, 0.32)
    ax.set_xlabel("previous-token attention score (non-circuit layer-0 heads)", fontsize=9, color=INK2)
    ax.set_ylabel("effect ÷ whole-circuit effect", fontsize=9, color=INK2)
    ax.set_title("Weak previous-token heads only matter once the main one is gone", fontsize=10, color=INK, loc="left")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    plt.setp(leg.get_texts(), color=INK)
    fig.text(0.01, 0.005, "Hollow = ablated alone; filled = after its strongest reference previous-token partner is ablated. "
             f"Mean ablation; values < {floor:g} drawn at {floor:g}.", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/q4")
    runs, excluded = load(d)
    s = summarize(runs)
    (d / "summary.json").write_text(json.dumps({"by_heads": {str(k): v for k, v in s.items()}, "excluded": excluded}, indent=1))
    write_markdown(runs, s, d / "summary.md", excluded)
    plot_nested(s, d / "nested.png")
    plot_pairs(s, d / "pairs.png")
    plot_hidden(s, d / "hidden.png")
    print((d / "summary.md").read_text())


if __name__ == "__main__":
    main()
