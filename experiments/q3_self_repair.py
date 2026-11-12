"""Q3: Self-repair. When a head is ablated, do other components compensate?

For one model (``--layers`` 2 or 3, ``--heads`` per layer, one seed), under mean and
resample ablation:

* **decomposition** (``src/ablation/decompose.py``) of every single-head ablation and of
  groups of circuit heads: the exact change in target logit difference split into the
  ablated heads' direct effect, every later head's response, and the final LayerNorm.
  Repeated with the attention-input LayerNorms frozen at their clean scale, so later heads
  can respond only to what changed, not by being rescaled.
* **loss**: total effect vs the effect of the direct path alone (path-specific ablation).
* **pair tests** for backup pairs (induction + induction; reference previous-token head +
  a sub-threshold "weak" previous-token head, the Q4 hidden backups): interaction in
  margin and loss with every LayerNorm free, the final one frozen, the attention-input
  ones frozen, or all frozen. If an interaction survives with all LayerNorms frozen it is
  static (parallel heads whose summed signal saturates a softmax), not compensation.

Models: 2 layers = the Q1 (8 heads) and Q4 (2 and 4 heads) checkpoints; 3 layers =
trained here to criterion with the Q1 recipe.

    python experiments/q3_self_repair.py --layers 3 --heads 8 --seed 0

Writes ``results/q3/L<layers>_h<heads>_seed<seed>.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import torch

from ablation import tasks as T
from ablation.ablate import AblationContext, effect_hooks, path_hooks, frozen_ln_hooks
from ablation.circuit import head_scores
from ablation.decompose import Decomposer
from ablation.model import Config, Transformer
from ablation.stats import bootstrap_ci, interaction_test
from ablation.train import TrainConfig, train
from q4_redundancy import meets_criterion, task_accuracies

METHODS = ("mean", "resample")
FREEZE = ("none", "final", "attn_inputs", "all")
WEAK_PREV = 0.1  # previous-token score of a "weak" (sub-threshold) previous-token head


def get_model(layers: int, heads: int, seed: int, out: Path, min_steps: int, max_steps: int) -> tuple[Transformer, dict]:
    if layers == 2:
        if heads == 8:
            m = Transformer.load(Path("results/q1/models") / f"seed{seed}.pt")
            return m, {"source": "q1 checkpoint", "steps": 1500, "met_criterion": meets_criterion(m)}
        path = Path("results/q4/models") / f"h{heads}_seed{seed}.pt"
        info = json.loads(path.with_suffix(".json").read_text())
        return Transformer.load(path), {**info, "source": "q4 checkpoint"}
    path = out / "models" / f"L{layers}_h{heads}_seed{seed}.pt"
    info_path = path.with_suffix(".json")
    if path.exists() and info_path.exists():
        return Transformer.load(path), json.loads(info_path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config(vocab_size=64, n_ctx=56, d_model=64, n_layers=layers, n_heads=heads, d_head=16, n_tasks=T.N_TAGS)
    model, log = train(cfg, T.Mixture(), TrainConfig(steps=max_steps, seed=seed), verbose=False,
                       stop_fn=meets_criterion, min_steps=min_steps, check_every=250)
    info = {"source": "trained to criterion", "steps": max(e["step"] for e in log) + 1,
            "met_criterion": meets_criterion(model), "accuracies": task_accuracies(model)}
    model.save(path)
    info_path.write_text(json.dumps(info, indent=1))
    return model, info


def roles(scores, n_layers: int, threshold: float) -> dict[tuple[int, int], str]:
    """Reference roles from attention patterns. Layer 0 (and any layer but the last):
    previous-token heads; layer >= 1: induction heads (induction wins if both)."""
    out = {}
    L, H = scores.prev_token.shape
    for l in range(L):
        for h in range(H):
            if l >= 1 and scores.induction[l, h] > threshold:
                out[(l, h)] = "ind"
            elif l < n_layers - 1 and scores.prev_token[l, h] > threshold:
                out[(l, h)] = "prev"
            else:
                out[(l, h)] = "non"
    return out


def key(h) -> str:
    return f"{h[0]}.{h[1]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/q3"))
    ap.add_argument("--steps", type=int, default=1500, help="minimum training steps (3-layer models)")
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=2048)
    ap.add_argument("--n-ref", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--circuit-threshold", type=float, default=0.3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"[L{args.layers} h{args.heads} seed {args.seed}]"

    model, train_info = get_model(args.layers, args.heads, args.seed, args.out, args.steps, args.max_steps)
    model.eval()
    print(f"{tag} model: {train_info} ({time.time() - t0:.0f}s)", flush=True)

    induction = T.Mixture().tasks[0]
    g = lambda k: torch.Generator().manual_seed(1_000_000 * (k + 1) + args.seed)  # same streams as Q1/Q4
    fit_gen = g(3)
    ctx = AblationContext(
        model, induction.sample(args.n_eval, g(0)), induction.sample(args.n_ref, g(1)),
        induction.sample(args.n_eval, g(2)), lambda n: induction.sample(n, fit_gen),
    )
    batch = ctx.eval_batch
    dec = Decomposer(ctx)
    scores = head_scores(model, batch)
    role = roles(scores, args.layers, args.circuit_threshold)
    L = args.layers
    with torch.no_grad():  # share of attention on the current position (layer-2 heads in 3-layer models)
        _, pc = model.run_with_cache(batch.tokens, names=[f"blocks.{l}.attn.pattern" for l in range(L)])
        w = batch.mask.float().unsqueeze(1)
        self_score = [((pc[f"blocks.{l}.attn.pattern"].diagonal(dim1=-2, dim2=-1) * w).sum((0, 2)) / w.sum((0, 2))).tolist()
                      for l in range(L)]
        del pc

    def summ(x) -> dict:
        m, lo, hi = bootstrap_ci(x.numpy())
        return {"mean": m, "lo": lo, "hi": hi}

    def decomp_record(heads, method, freeze=None) -> dict:
        d = dec(list(heads), method, freeze_ln=freeze)
        resp = {key(h): float(v.mean()) for h, v in d.responses.items()}
        return {
            "total": summ(d.total), "direct": summ(d.direct), "responses": summ(d.response_sum),
            "ln_final": summ(d.ln_final), "repair": summ(d.total - d.direct),
            "loss": float(d.loss.mean()), "margin_drop": float(d.margin_drop.mean()),
            "response_by_head": resp,
        }

    @torch.no_grad()
    def run_margins(heads, method, freeze="none"):
        hooks = effect_hooks(ctx, list(heads), method, freeze_ln=False if freeze == "none" else freeze)
        return T.margins(model(batch.tokens, hooks=hooks), batch)

    # self-check: heads that cannot be affected do not change
    check = max(dec.unaffected_change([h], "mean") for h in ctx.all_heads[: args.heads * (L - 1)])

    # 1. single heads
    single = {}
    for h in ctx.all_heads:
        rec = {"role": role[h]}
        for method in METHODS:
            r = decomp_record([h], method)
            if h[0] < L - 1:
                r["frozen_attn_ln"] = decomp_record([h], method, "attn_inputs")
                with torch.no_grad():
                    lg = model(batch.tokens, hooks=path_hooks(ctx, [h], method, ["direct"]))
                r["loss_direct_path"] = float((T.per_seq_loss(lg, batch) - ctx.clean_loss).mean())
            else:
                r["loss_direct_path"] = r["loss"]
            rec[method] = r
        single[key(h)] = rec
    print(f"{tag} singles done ({time.time() - t0:.0f}s)", flush=True)

    # 2. groups of circuit heads
    by_role = lambda r, l=None: [h for h in ctx.all_heads if role[h] == r and (l is None or h[0] == l)]
    group_defs = {"prev_token_heads": by_role("prev"), "induction_heads": by_role("ind")}
    for l in range(1, L):
        group_defs[f"induction_heads_L{l}"] = by_role("ind", l)
        group_defs[f"layer{l}"] = [h for h in ctx.all_heads if h[0] == l]
    groups = {}
    for name, hs in group_defs.items():
        if hs and name not in groups and not (name.startswith("induction_heads_L") and hs == group_defs["induction_heads"]):
            groups[name] = {"heads": [key(h) for h in hs], **{m: decomp_record(hs, m) for m in METHODS}}
    print(f"{tag} groups done ({time.time() - t0:.0f}s)", flush=True)

    # 3. pair tests under LayerNorm freezing
    prev = by_role("prev")
    weak = [h for h in ctx.all_heads if role[h] == "non" and h[0] < L - 1 and scores.prev_token[h[0], h[1]] >= WEAK_PREV]
    pair_list = [("ind-ind", a, b) for a, b in itertools.combinations(by_role("ind"), 2)]
    pair_list += [("prev-weak", p, w) for w in weak for p in prev if p[0] == w[0]]
    margin_cache = {}

    def margins_of(heads, method, freeze):
        k = (tuple(sorted(heads)), method, freeze)
        if k not in margin_cache:
            margin_cache[k] = run_margins(heads, method, freeze)
        return margin_cache[k]

    pairs = []
    for kind, a, b in pair_list:
        row = {"kind": kind, "a": key(a), "b": key(b), "same_layer": a[0] == b[0]}
        for method in METHODS:
            row[method] = {}
            for fz in FREEZE:
                m0 = dec.m_clean
                row[method][fz] = interaction_test(m0, margins_of([a], method, fz), margins_of([b], method, fz),
                                                   margins_of([a, b], method, fz), batch.mask)
        pairs.append(row)
    print(f"{tag} {len(pairs)} pairs done ({time.time() - t0:.0f}s)", flush=True)

    out = {
        "layers": L, "heads": args.heads, "seed": args.seed,
        "training": train_info,
        "config": model.cfg.to_dict(),
        "clean_eval_loss": ctx.clean_loss.mean().item(),
        "clean_logit_diff": float(dec.ld_clean.mean()),
        "unaffected_change_check": check,
        "head_scores": {**scores.to_dict(), "self": self_score},
        "roles": {key(h): r for h, r in role.items()},
        "single": single,
        "groups": groups,
        "pairs": pairs,
        "runtime_s": round(time.time() - t0, 1),
    }
    path = args.out / f"L{L}_h{args.heads}_seed{args.seed}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"{tag} wrote {path} in {out['runtime_s']}s", flush=True)


if __name__ == "__main__":
    main()
