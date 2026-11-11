"""Q2 follow-up: which LayerNorm cushions ablations, and is the interaction loss curvature?

For one seed, for every single head and the Q2 head groups:

1. zero / mean / shift ablation with only the final LayerNorm frozen, and with only the
   LayerNorms in front of attention frozen (Q2 froze all of them at once);
2. the margin test: per-position margins m = z_target - logsumexp(others) under clean,
   mean, shift and zero ablation. If the margin drops of mean and shift add up to zero's,
   the superadditive *loss* interaction found in Q2 comes purely from the convexity of
   cross-entropy (softplus(-m)) in a saturated model.

    python experiments/q2_followup.py --seed 0

Writes ``results/q2/followup_seed<seed>.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ablation import tasks as T
from ablation.ablate import AblationContext, effect_hooks
from ablation.model import Transformer
from ablation.stats import bootstrap_ci, interaction_test

METHODS = ("zero", "mean", "shift")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--q1", type=Path, default=Path("results/q1"))
    ap.add_argument("--out", type=Path, default=Path("results/q2"))
    ap.add_argument("--n-eval", type=int, default=2048)
    ap.add_argument("--n-ref", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()

    model = Transformer.load(args.q1 / "models" / f"seed{args.seed}.pt")
    q1 = json.loads((args.q1 / f"seed{args.seed}.json").read_text())
    task = T.InductionTask()
    g = lambda k: torch.Generator().manual_seed(1_000_000 * (k + 1) + args.seed)  # same streams as Q1/Q2
    fit_gen = g(3)
    ctx = AblationContext(
        model, task.sample(args.n_eval, g(0)), task.sample(args.n_ref, g(1)), task.sample(args.n_eval, g(2)),
        lambda n: task.sample(n, fit_gen),
    )
    batch = ctx.eval_batch

    @torch.no_grad()
    def run(hooks):
        logits = model(batch.tokens, hooks=hooks)
        return T.per_seq_loss(logits, batch) - ctx.clean_loss, T.margins(logits, batch)

    _, m_clean = run({})
    circuit = [tuple(h) for h in q1["reference_circuit"]]
    all_heads = ctx.all_heads
    groups = {
        "non_circuit": [h for h in all_heads if h not in circuit],
        "prev_token_heads": [h for h in circuit if h[0] == 0],
        "induction_heads": [h for h in circuit if h[0] >= 1],
        "layer0": [h for h in all_heads if h[0] == 0],
    }
    units = {f"{l}.{h}": [(l, h)] for l, h in all_heads} | {f"group:{k}": v for k, v in groups.items() if v}

    results = {}
    for name, heads in units.items():
        entry: dict = {}
        margins = {}
        for method in METHODS:
            eff, margins[method] = run(effect_hooks(ctx, heads, method))
            entry[method] = bootstrap_ci(eff.numpy())[0]
            for which in ("final", "attn_inputs"):
                e, _ = run(effect_hooks(ctx, heads, method, freeze_ln=which))
                entry[f"{method}_frozen_{which}"] = bootstrap_ci(e.numpy())[0]
        entry["margin_test"] = interaction_test(m_clean, margins["mean"], margins["shift"], margins["zero"], batch.mask)
        results[name] = entry
        mt = entry["margin_test"]
        print(f"[seed {args.seed}] {name:24s} zero={entry['zero']:.4f} final={entry['zero_frozen_final']:.4f} "
              f"attn={entry['zero_frozen_attn_inputs']:.4f} | loss int={mt['loss_interaction']:.4f} "
              f"margin int={mt['margin_interaction']:.3f} explained={mt['explained']:.2f}", flush=True)

    m = batch.mask
    out = {
        "seed": args.seed,
        "reference_circuit": [list(h) for h in circuit],
        "clean_margin": {
            "mean": float(m_clean[m].mean()), "median": float(m_clean[m].median()),
            "p05": float(m_clean[m].quantile(0.05)),
        },
        "units": results,
        "runtime_s": round(time.time() - t0, 1),
    }
    path = args.out / f"followup_seed{args.seed}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"[seed {args.seed}] wrote {path} in {out['runtime_s']}s", flush=True)


if __name__ == "__main__":
    main()
