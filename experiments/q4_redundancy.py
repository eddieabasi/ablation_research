"""Q4: Does single-component ablation miss components that only matter jointly?

For one model (``--heads`` heads per layer, one seed):

* single-head effects, for every head
* every pair of heads ablated jointly, with the interaction
  ``effect(A and B) - effect(A) - effect(B)`` (per-sequence bootstrap CI) and the margin
  test (does the interaction exist in margin space, or only in the loss?)
* nested ablation of the induction heads: remove k of them (random subsets) for every k
* sufficiency: keep exactly one induction head (or one previous-token head), ablate the
  others of that kind

Methods: mean ablation (primary) and resample ablation. 8-head models are the Q1
checkpoints; other head counts are trained here with the same recipe, *to criterion*:
at least 1,500 steps (the Q1 budget), then until both tasks reach 99% held-out accuracy,
up to ``--max-steps``. Small models sit on the induction plateau much longer than 8-head
ones; with a fixed 1,500 steps most of them never learned induction at all. Models that
miss the criterion are still analysed but flagged (``met_criterion``).

    python experiments/q4_redundancy.py --heads 4 --seed 0

Writes ``results/q4/h<heads>_seed<seed>.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import torch

from ablation import tasks as T
from ablation.ablate import AblationContext, effect_hooks
from ablation.circuit import head_scores, reference_circuit
from ablation.model import Config, Transformer
from ablation.stats import bootstrap_ci, interaction_test
from ablation.train import TrainConfig, train

METHODS = ("mean", "resample")
MAX_SUBSETS = 10


CRITERION = 0.99


@torch.no_grad()
def task_accuracies(model: Transformer, n: int = 512) -> dict[str, float]:
    """Held-out accuracy on each task (generator independent of training and evaluation data)."""
    gen = torch.Generator().manual_seed(424242)
    induction, two_back = T.Mixture().tasks
    out = {}
    for name, task in (("induction", induction), ("two_back", two_back)):
        b = task.sample(n, gen)
        out[name] = T.accuracy(model(b.tokens), b)
    return out


def meets_criterion(model: Transformer) -> bool:
    return all(a >= CRITERION for a in task_accuracies(model).values())


def get_model(heads: int, seed: int, q1: Path, out: Path, steps: int, max_steps: int) -> tuple[Transformer, dict]:
    if heads == 8 and (q1 / "models" / f"seed{seed}.pt").exists():
        m = Transformer.load(q1 / "models" / f"seed{seed}.pt")
        return m, {"source": "q1 checkpoint", "steps": steps, "met_criterion": meets_criterion(m)}
    path = out / "models" / f"h{heads}_seed{seed}.pt"
    info_path = path.with_suffix(".json")
    if path.exists() and info_path.exists():
        return Transformer.load(path), json.loads(info_path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config(vocab_size=64, n_ctx=56, d_model=64, n_layers=2, n_heads=heads, d_head=16, n_tasks=T.N_TAGS)
    model, log = train(cfg, T.Mixture(), TrainConfig(steps=max_steps, seed=seed), verbose=False,
                       stop_fn=meets_criterion, min_steps=steps, check_every=250)
    trained = max(e["step"] for e in log) + 1
    info = {"source": "trained to criterion", "steps": trained, "met_criterion": meets_criterion(model),
            "accuracies": task_accuracies(model)}
    model.save(path)
    info_path.write_text(json.dumps(info, indent=1))
    return model, info


def pair_kind(a, b, circuit) -> str:
    def k(h):
        if h not in circuit:
            return "non"
        return "prev" if h[0] == 0 else "ind"

    kinds = sorted([k(a), k(b)])
    return {
        ("ind", "ind"): "ind-ind",
        ("prev", "prev"): "prev-prev",
        ("ind", "prev"): "prev-ind",
        ("non", "non"): "non-non",
    }.get(tuple(kinds), "circuit-non")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--q1", type=Path, default=Path("results/q1"))
    ap.add_argument("--out", type=Path, default=Path("results/q4"))
    ap.add_argument("--steps", type=int, default=1500, help="minimum training steps (Q1 budget)")
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=2048)
    ap.add_argument("--n-ref", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--circuit-threshold", type=float, default=0.3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"[h{args.heads} seed {args.seed}]"

    model, train_info = get_model(args.heads, args.seed, args.q1, args.out, args.steps, args.max_steps)
    model.eval()
    print(f"{tag} model: {train_info} ({time.time() - t0:.0f}s)", flush=True)

    induction, two_back = T.Mixture().tasks
    g = lambda k: torch.Generator().manual_seed(1_000_000 * (k + 1) + args.seed)  # same streams as Q1
    fit_gen = g(3)
    ctx = AblationContext(
        model, induction.sample(args.n_eval, g(0)), induction.sample(args.n_ref, g(1)),
        induction.sample(args.n_eval, g(2)), lambda n: induction.sample(n, fit_gen),
    )
    batch = ctx.eval_batch
    with torch.no_grad():
        clean = {}
        for name, task in (("induction", induction), ("two_back", two_back)):
            b = task.sample(args.n_eval, g(4))
            lg = model(b.tokens)
            clean[name] = {"loss": T.loss(lg, b).item(), "acc": T.accuracy(lg, b)}
        m_clean = T.margins(model(batch.tokens), batch)

    scores = head_scores(model, batch)
    circuit = sorted(reference_circuit(scores, args.circuit_threshold))
    cset = set(circuit)
    all_heads = ctx.all_heads
    ind_heads = [h for h in circuit if h[0] >= 1]
    prev_heads = [h for h in circuit if h[0] == 0]

    @torch.no_grad()
    def run(heads, method):
        logits = model(batch.tokens, hooks=effect_hooks(ctx, list(heads), method))
        return T.per_seq_loss(logits, batch) - ctx.clean_loss, T.margins(logits, batch)

    def summ(eff):
        m, lo, hi = bootstrap_ci(eff.numpy())
        return {"effect": m, "lo": lo, "hi": hi}

    # singles
    single, single_cache = {}, {}
    for h in all_heads:
        key = f"{h[0]}.{h[1]}"
        single[key] = {}
        for method in METHODS:
            eff, mg = run([h], method)
            single_cache[(h, method)] = (eff, mg)
            single[key][method] = summ(eff)
    print(f"{tag} singles done ({time.time() - t0:.0f}s)", flush=True)

    # pairs
    pairs = []
    for a, b in itertools.combinations(all_heads, 2):
        row = {"a": list(a), "b": list(b), "kind": pair_kind(a, b, cset)}
        for method in METHODS:
            eff_ab, m_ab = run([a, b], method)
            eff_a, m_a = single_cache[(a, method)]
            eff_b, m_b = single_cache[(b, method)]
            inter = summ(eff_ab - eff_a - eff_b)
            row[method] = {
                "effect_AB": float(eff_ab.mean()),
                "interaction": inter["effect"], "interaction_lo": inter["lo"], "interaction_hi": inter["hi"],
                "margin_test": interaction_test(m_clean, m_a, m_b, m_ab, batch.mask),
            }
        pairs.append(row)
    print(f"{tag} {len(pairs)} pairs done ({time.time() - t0:.0f}s)", flush=True)

    # groups, nested curves and sufficiency (mean ablation)
    groups = {}
    for name, heads in (("circuit", circuit), ("induction_heads", ind_heads), ("prev_token_heads", prev_heads),
                        ("layer1", [h for h in all_heads if h[0] == 1])):
        if heads:
            groups[name] = {"heads": [list(h) for h in heads], **{m: summ(run(heads, m)[0]) for m in METHODS}}

    rng = random.Random(args.seed)
    nested = {}
    for name, heads in (("induction_heads", ind_heads), ("prev_token_heads", prev_heads)):
        if not heads:
            continue
        curve = []
        for k in range(1, len(heads) + 1):
            combos = list(itertools.combinations(heads, k))
            chosen = combos if len(combos) <= MAX_SUBSETS else rng.sample(combos, MAX_SUBSETS)
            effs = [float(run(c, "mean")[0].mean()) for c in chosen]
            curve.append({"k": k, "n_subsets": len(chosen), "effects": effs})
        nested[name] = curve

    sufficiency = {}
    for name, heads in (("induction_heads", ind_heads), ("prev_token_heads", prev_heads)):
        if len(heads) < 2:
            continue
        sufficiency[name] = {
            f"{h[0]}.{h[1]}": float(run([o for o in heads if o != h], "mean")[0].mean()) for h in heads
        }
    print(f"{tag} groups/nested/sufficiency done ({time.time() - t0:.0f}s)", flush=True)

    out = {
        "heads": args.heads,
        "seed": args.seed,
        "training": train_info,
        "config": model.cfg.to_dict(),
        "clean": clean,
        "clean_eval_loss": ctx.clean_loss.mean().item(),
        "chance_loss": induction.chance_loss,
        "head_scores": scores.to_dict(),
        "reference_circuit": [list(h) for h in circuit],
        "single": single,
        "pairs": pairs,
        "groups": groups,
        "nested": nested,
        "sufficiency": sufficiency,
        "runtime_s": round(time.time() - t0, 1),
    }
    path = args.out / f"h{args.heads}_seed{args.seed}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"{tag} wrote {path} in {out['runtime_s']}s", flush=True)


if __name__ == "__main__":
    main()
