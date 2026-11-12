"""Aggregate Q3 results (results/q3/L<layers>_h<heads>_seed<k>.json) into summary.md / summary.json and figures.

    python experiments/q3_analyze.py results/q3

All decomposition numbers are changes in target logit difference (logits), mean ablation
unless stated. Sign convention: negative = less logit on the right answer (damage);
positive = more (repair).
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
# Reference palette slots 1-3 (validated all-pairs, light mode), one per component.
C_DIRECT, C_RESP, C_LN = "#2a78d6", "#eb6834", "#1baf7a"
MIN_DAMAGE = 0.1  # logits; heads with less direct (or gross) damage are left out of ratios
CLASSES = [(2, 2), (2, 4), (2, 8), (3, 4), (3, 8)]
FREEZE = ("none", "final", "attn_inputs", "all")


def label(c) -> str:
    return f"{c[0]} layers, {c[1]} heads"


def short(c) -> str:
    return f"{c[0]}L × {c[1]}h"


def load(d: Path) -> dict[tuple[int, int], list[dict]]:
    runs: dict = defaultdict(list)
    for p in sorted(d.glob("L*_h*_seed*.json")):
        r = json.loads(p.read_text())
        if r["training"].get("met_criterion", True):
            runs[(r["layers"], r["heads"])].append(r)
    return {c: runs[c] for c in CLASSES if c in runs}


def med(xs, d=2) -> str:
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    return f"{np.median(xs):.{d}f}" if len(xs) else "–"


def fmt(xs, d=2) -> str:
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    if len(xs) == 0:
        return "–"
    return f"{xs.mean():.{d}f} ± {xs.std(ddof=1):.{d}f}" if len(xs) > 1 else f"{xs.mean():.{d}f}"


def hl(k: str) -> int:
    return int(k.split(".")[0])


def components(rec: dict, L: int) -> dict:
    """Mean components of one decomposition record, with responses split by layer."""
    out = {"direct": rec["direct"]["mean"], "ln": rec["ln_final"]["mean"], "total": rec["total"]["mean"]}
    by_layer = defaultdict(float)
    pos = neg = 0.0
    for k, v in rec["response_by_head"].items():
        by_layer[hl(k)] += v
        pos += max(v, 0.0)
        neg += min(v, 0.0)
    out["resp"] = rec["responses"]["mean"]
    out["resp_by_layer"] = dict(by_layer)
    out["resp_pos"], out["resp_neg"] = pos, neg
    parts = [out["direct"], out["ln"]] + list(rec["response_by_head"].values())
    out["damage"] = -sum(p for p in parts if p < 0)
    out["repair"] = sum(p for p in parts if p > 0)
    return out


def summarize(runs) -> dict:
    S: dict = {}
    for c, rs in runs.items():
        L = c[0]
        o: dict = {"n_models": len(rs), "seeds": [r["seed"] for r in rs]}
        o["n_prev"] = [sum(v == "prev" for v in r["roles"].values()) for r in rs]
        o["n_ind"] = [sum(v == "ind" for v in r["roles"].values()) for r in rs]
        o["clean_logit_diff"] = [r["clean_logit_diff"] for r in rs]
        o["check"] = max(r["unaffected_change_check"] for r in rs)

        # single heads by role
        for method in ("mean", "resample"):
            rows = []
            for r in rs:
                for k, rec in r["single"].items():
                    cpt = components(rec[method], L)
                    row = {"seed": r["seed"], "head": k, "layer": hl(k), "role": rec["role"], **cpt,
                           "loss": rec[method]["loss"], "loss_direct_path": rec[method]["loss_direct_path"]}
                    if "frozen_attn_ln" in rec[method]:
                        f = components(rec[method]["frozen_attn_ln"], L)
                        row["frozen_resp"], row["frozen_total"] = f["resp"], f["total"]
                    rows.append(row)
            o[f"single_{method}"] = rows

        # groups
        g = {}
        for name in ("induction_heads", "prev_token_heads", "layer1", "layer2", "induction_heads_L1", "induction_heads_L2"):
            vals = []
            for r in rs:
                if name not in r["groups"]:
                    continue
                cpt = components(r["groups"][name]["mean"], L)
                singles = [components(r["single"][k]["mean"], L) for k in r["groups"][name]["heads"]]
                cpt["sum_single_total"] = sum(s["total"] for s in singles)
                cpt["sum_single_ln"] = sum(s["ln"] for s in singles)
                cpt["sum_single_resp"] = sum(s["resp"] for s in singles)
                cpt["sum_single_direct"] = sum(s["direct"] for s in singles)
                cpt["loss"] = r["groups"][name]["mean"]["loss"]
                cpt["n_heads"] = len(r["groups"][name]["heads"])
                vals.append(cpt)
            if vals:
                g[name] = vals
        o["groups"] = g

        # pairs under LayerNorm freezing
        pk = {}
        for method in ("mean", "resample"):
            pk[method] = {}
            for kind in ("ind-ind", "prev-weak"):
                rows = [p for r in rs for p in r["pairs"] if p["kind"] == kind and p["same_layer"]]
                if rows:
                    pk[method][kind] = {fz: {"margin": [p[method][fz]["margin_interaction"] for p in rows],
                                             "loss": [p[method][fz]["loss_interaction"] for p in rows]} for fz in FREEZE}
                    pk[method][kind]["n"] = len(rows)
        o["pairs"] = pk
        S[c] = o
    return S


def write_markdown(S: dict, path: Path) -> None:
    L_ = ["# Q3 summary\n",
          "Changes in target logit difference (logits) from mean ablation unless stated; negative = damage, "
          "positive = repair. *Direct* = the ablated head's own change in contribution; *responses* = change in "
          "every later head's contribution; *LN* = the final LayerNorm's rescaling. total = direct + responses + LN "
          f"exactly. Ratios use heads whose direct (or gross) damage is at least {MIN_DAMAGE} logits.\n"]
    L_.append("## Models\n")
    L_.append("| model | seeds | prev-token heads | induction heads | clean logit diff | self-check (max change of unaffected heads) |")
    L_.append("|---|---|---|---|---|---|")
    for c, o in S.items():
        L_.append(f"| {label(c)} | {o['n_models']} ({', '.join(map(str, o['seeds']))}) | {fmt(o['n_prev'], 1)} | "
                  f"{fmt(o['n_ind'], 1)} | {fmt(o['clean_logit_diff'], 1)} | {o['check']:.1e} |")

    # 1. induction heads: who repairs a single induction head's direct damage
    L_.append("\n## 1. One induction head ablated: how much of its direct damage is repaired, and by what\n")
    L_.append("Shares of the head's direct damage (−direct). LN = final LayerNorm; responses = heads in later layers "
              "(3-layer models: layer 2). Repaired = (total − direct) ÷ −direct.\n")
    L_.append("| model | method | heads | median direct | LN share | responses share | repaired | heads with repaired > 0 |")
    L_.append("|---|---|---|---|---|---|---|---|")
    for method in ("mean", "resample"):
        for c, o in S.items():
            rows = [x for x in o[f"single_{method}"] if x["role"] == "ind" and -x["direct"] >= MIN_DAMAGE]
            if not rows:
                continue
            dd = np.array([-x["direct"] for x in rows])
            ln = np.array([x["ln"] for x in rows]) / dd
            rp = np.array([x["resp"] for x in rows]) / dd
            rep = np.array([x["total"] - x["direct"] for x in rows]) / dd
            L_.append(f"| {label(c)} | {method} | {len(rows)} | {med([x['direct'] for x in rows])} | {med(ln)} | {med(rp)} | "
                      f"{med(rep)} | {np.mean(rep > 0):.0%} |")

    # 2. previous-token heads
    L_.append("\n## 2. One previous-token head ablated: the damage is indirect\n")
    L_.append("Responses split by layer. LN1 cushion = share of the responses' damage that the attention-input "
              "LayerNorms prevent: 1 − responses ÷ responses with those LayerNorms frozen.\n")
    L_.append("| model | heads | median direct | median responses: layer 1 | median responses: layer 2 | median LN | median total | median LN1 cushion |")
    L_.append("|---|---|---|---|---|---|---|---|")
    for c, o in S.items():
        rows = [x for x in o["single_mean"] if x["role"] == "prev"]
        if not rows:
            continue
        cush = [1 - x["resp"] / x["frozen_resp"] for x in rows if "frozen_resp" in x and x["frozen_resp"] < -MIN_DAMAGE]
        L_.append(f"| {label(c)} | {len(rows)} | {med([x['direct'] for x in rows])} | "
                  f"{med([x['resp_by_layer'].get(1, 0.0) for x in rows])} | "
                  f"{med([x['resp_by_layer'].get(2, np.nan) for x in rows]) if c[0] == 3 else '–'} | "
                  f"{med([x['ln'] for x in rows])} | {med([x['total'] for x in rows])} | {med(cush)} |")

    # 3. repair ratio for every head with real damage, and its source
    L_.append("\n## 3. Every head: gross damage vs repair, and where the repair comes from\n")
    L_.append("Gross damage D = sum of negative components (direct, each response, LN); repair R = sum of positive "
              "ones. Source shares are shares of total R summed over heads.\n")
    L_.append("| model | method | heads with D ≥ 0.1 | median R ÷ D | R from final LN | R from later heads | R from the head itself |")
    L_.append("|---|---|---|---|---|---|---|")
    for method in ("mean", "resample"):
        for c, o in S.items():
            rows = [x for x in o[f"single_{method}"] if x["damage"] >= MIN_DAMAGE]
            if not rows:
                continue
            R_ln = sum(max(x["ln"], 0) for x in rows)
            R_resp = sum(x["resp_pos"] for x in rows)
            R_dir = sum(max(x["direct"], 0) for x in rows)
            tot = R_ln + R_resp + R_dir
            L_.append(f"| {label(c)} | {method} | {len(rows)} | {med([x['repair'] / x['damage'] for x in rows])} | "
                      f"{R_ln / tot:.0%} | {R_resp / tot:.0%} | {R_dir / tot:.0%} |")

    # 4. mean vs resample
    L_.append("\n## 4. Mean vs resample ablation\n")
    L_.append("Per head (D ≥ 0.1 under both): R ÷ D under each method.\n")
    L_.append("| model | heads | median R ÷ D, mean | median R ÷ D, resample | share of heads where mean > resample |")
    L_.append("|---|---|---|---|---|")
    for c, o in S.items():
        rm = {(x["seed"], x["head"]): x for x in o["single_mean"]}
        rr = {(x["seed"], x["head"]): x for x in o["single_resample"]}
        ks = [k for k in rm if rm[k]["damage"] >= MIN_DAMAGE and rr[k]["damage"] >= MIN_DAMAGE]
        if not ks:
            continue
        a = np.array([rm[k]["repair"] / rm[k]["damage"] for k in ks])
        b = np.array([rr[k]["repair"] / rr[k]["damage"] for k in ks])
        L_.append(f"| {label(c)} | {len(ks)} | {np.median(a):.2f} | {np.median(b):.2f} | {np.mean(a > b):.0%} |")

    # 5. groups: superadditivity in logit space, attributed exactly
    L_.append("\n## 5. Whole groups: repair that works for one head fails for all of them\n")
    L_.append("Logit-diff interaction = total(group) − Σ total(single heads). Because direct effects add exactly, "
              "the interaction is exactly (LN(group) − Σ LN(singles)) + (responses(group) − Σ responses(singles)).\n")
    L_.append("| model | group | heads | total (group) | Σ totals (singles) | interaction | from LN | from responses | LN share of direct: singles → group |")
    L_.append("|---|---|---|---|---|---|---|---|---|")
    for c, o in S.items():
        for name, vals in o["groups"].items():
            if name.startswith("induction_heads_L") or name == "layer2":
                continue
            inter = [v["total"] - v["sum_single_total"] for v in vals]
            f_ln = [v["ln"] - v["sum_single_ln"] for v in vals]
            f_rs = [v["resp"] - v["sum_single_resp"] for v in vals]
            sh_s = [v["sum_single_ln"] / -v["sum_single_direct"] for v in vals if -v["sum_single_direct"] > MIN_DAMAGE]
            sh_g = [v["ln"] / -v["direct"] for v in vals if -v["direct"] > MIN_DAMAGE]
            L_.append(f"| {label(c)} | {name} | {fmt([v['n_heads'] for v in vals], 1)} | {fmt([v['total'] for v in vals], 1)} | "
                      f"{fmt([v['sum_single_total'] for v in vals], 1)} | {fmt(inter, 1)} | {fmt(f_ln, 1)} | {fmt(f_rs, 1)} | "
                      f"{fmt(sh_s, 2)} → {fmt(sh_g, 2)} |")

    # 6. pairs under LayerNorm freezing
    L_.append("\n## 6. Backup pairs with LayerNorms frozen: static redundancy or compensation?\n")
    L_.append("Margin interaction (nats; > 0 superadditive) of same-layer pairs, median over pairs, with every "
              "LayerNorm free, the final one frozen, the attention-input ones frozen, or all frozen. Kept = Σ "
              "interaction frozen ÷ Σ interaction free (pooled over pairs).\n")
    L_.append("| model | method | pair type | n | free | final frozen | attn-input frozen | all frozen | kept: final | kept: attn-input | kept: all |")
    L_.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for method in ("mean", "resample"):
        for c, o in S.items():
            for kind, d in o["pairs"].get(method, {}).items():
                base = np.sum(d["none"]["margin"])
                kept = lambda fz: f"{np.sum(d[fz]['margin']) / base:.0%}" if abs(base) > 1e-9 else "–"
                L_.append(f"| {label(c)} | {method} | {kind} | {d['n']} | {med(d['none']['margin'], 3)} | "
                          f"{med(d['final']['margin'], 3)} | {med(d['attn_inputs']['margin'], 3)} | {med(d['all']['margin'], 3)} | "
                          f"{kept('final')} | {kept('attn_inputs')} | {kept('all')} |")

    # 7. loss: total vs direct path
    L_.append("\n## 7. Loss: total effect vs direct path only\n")
    L_.append("Heads not in the last layer. Direct path = the head's change delivered only to the unembedding "
              "(path-specific ablation). Total < direct path means the rest of the network repairs more than it adds.\n")
    L_.append("| model | role | heads | median loss: total | median loss: direct path | share with total < direct path |")
    L_.append("|---|---|---|---|---|---|")
    for c, o in S.items():
        for rl in ("prev", "ind", "non"):
            rows = [x for x in o["single_mean"] if x["role"] == rl and x["layer"] < c[0] - 1]
            if not rows:
                continue
            L_.append(f"| {label(c)} | {rl} | {len(rows)} | {med([x['loss'] for x in rows], 3)} | "
                      f"{med([x['loss_direct_path'] for x in rows], 3)} | "
                      f"{np.mean([x['loss'] < x['loss_direct_path'] for x in rows]):.0%} |")
    path.write_text("\n".join(L_) + "\n")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def plot_decomposition(S: dict, path: Path) -> None:
    """Induction heads: direct damage normalised to −1, with LN and later-head responses stacked."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200, facecolor=SURFACE, sharey=True)
    for ax, mode in zip(axes, ("single", "group")):
        _style(ax)
        xs, labels = [], []
        for i, (c, o) in enumerate(S.items()):
            if mode == "single":
                rows = [x for x in o["single_mean"] if x["role"] == "ind" and -x["direct"] >= MIN_DAMAGE]
                if not rows:
                    continue
                dd = np.array([-x["direct"] for x in rows])
                ln = np.median(np.array([x["ln"] for x in rows]) / dd)
                rp = np.median(np.array([x["resp"] for x in rows]) / dd)
                tot = np.median(np.array([x["total"] for x in rows]) / dd)
            else:
                vals = o["groups"].get("induction_heads")
                if not vals:
                    continue
                ln = np.median([v["ln"] / -v["direct"] for v in vals])
                rp = np.median([v["resp"] / -v["direct"] for v in vals])
                tot = np.median([v["total"] / -v["direct"] for v in vals])
            w = 0.6
            ax.bar(i, -1, w, color=C_DIRECT, label="direct (the head's own output)" if not xs else None, zorder=2)
            # positive parts stack upward from 0, negative parts stack below −1
            up, down = 0.0, -1.0
            for v, col, lab in ((ln, C_LN, "final LayerNorm"), (rp, C_RESP, "later heads' responses")):
                if v >= 0:
                    ax.bar(i, v, w, bottom=up, color=col, label=lab if not xs else None, zorder=2)
                    up += v
                else:
                    ax.bar(i, v, w, bottom=down, color=col, label=lab if not xs else None, zorder=2)
                    down += v
            ax.plot([i - w / 2, i + w / 2], [tot, tot], color=INK, linewidth=2, zorder=3,
                    label="net effect" if not xs else None)
            xs.append(i)
            labels.append(short(c))
        ax.set_xticks(xs, labels, fontsize=8, color=INK)
        ax.axhline(0, color=INK2, linewidth=1, zorder=1)
        ax.set_title("One induction head ablated" if mode == "single" else "All induction heads ablated",
                     fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("change in target logit ÷ direct damage", fontsize=8, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, frameon=False, fontsize=7.5, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.93))
    plt.setp(leg.get_texts(), color=INK)
    fig.suptitle("The final LayerNorm repairs a lost induction head, but not a lost group", fontsize=10, color=INK,
                 x=0.01, ha="left")
    fig.text(0.01, 0.005, "Mean ablation; medians over heads (left) or models (right). "
             "Bars: components of the exact decomposition; black line: net change.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_freeze(S: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), dpi=200, facecolor=SURFACE)
    variants = {"ind-ind": ("none", "final", "all"), "prev-weak": ("none", "attn_inputs", "all")}
    vlabels = {"none": "all free", "final": "final LN\nfrozen", "attn_inputs": "attention-input\nLNs frozen",
               "all": "all LNs\nfrozen"}
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd8"]
    for ax, kind, title in ((axes[0], "ind-ind", "Induction + induction (same layer)"),
                            (axes[1], "prev-weak", "Previous-token + weak previous-token")):
        _style(ax)
        vs = variants[kind]
        n = 0
        for ci, (c, o) in enumerate(S.items()):
            d = o["pairs"].get("mean", {}).get(kind)
            if not d:
                continue
            base = np.array(d["none"]["margin"])
            ys = [np.sum(d[v]["margin"]) / np.sum(base) for v in vs]
            ax.plot(range(len(vs)), ys, color=colors[ci % len(colors)], marker="o", markersize=5, linewidth=1.8,
                    markeredgecolor=SURFACE, label=f"{short(c)} (n={d['n']})", zorder=3)
            n += 1
        ax.axhline(0, color=INK2, linewidth=1, zorder=1)
        ax.set_xticks(range(len(vs)), [vlabels[v] for v in vs], fontsize=7.5, color=INK)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        if n:
            leg = ax.legend(frameon=False, fontsize=7.5, loc="upper right")
            plt.setp(leg.get_texts(), color=INK)
    axes[0].set_ylabel("margin interaction kept\n(Σ frozen ÷ Σ free)", fontsize=8, color=INK2)
    fig.suptitle("Backup superadditivity disappears when LayerNorm can no longer rescale", fontsize=10, color=INK,
                 x=0.01, ha="left")
    fig.text(0.01, 0.005, "Mean ablation; pooled over all same-layer pairs of each type.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/q3")
    runs = load(d)
    if not runs:
        sys.exit(f"no usable results in {d}")
    S = summarize(runs)
    (d / "summary.json").write_text(json.dumps({short(c): o for c, o in S.items()}, indent=1))
    write_markdown(S, d / "summary.md")
    plot_decomposition(S, d / "decomposition.png")
    plot_freeze(S, d / "freeze.png")
    print((d / "summary.md").read_text())


if __name__ == "__main__":
    main()
