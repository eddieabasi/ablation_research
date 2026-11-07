"""Q1: Do zero, mean, resample and optimal ablation agree on which heads matter?

For one seed:
1. Train a 2-layer attention-only transformer on a 50/50 mixture of induction and
   two-back sequences (task embedding at every position).
2. Identify the reference induction circuit from attention patterns alone
   (layer-0 previous-token heads, layer-1 induction heads).
3. Measure the loss increase on induction sequences from ablating each head alone and
   several head groups jointly, under each ablation method.

Writes ``<out>/seed<seed>.json`` and the model checkpoint ``<out>/models/seed<seed>.pt``.

    python experiments/q1_method_agreement.py --seed 0 --out results/q1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ablation import tasks as T
from ablation.ablate import METHODS, AblationContext, ablation_effect
from ablation.circuit import head_scores, reference_circuit
from ablation.model import Config, Transformer
from ablation.stats import bootstrap_ci
from ablation.train import TrainConfig, train


def model_config() -> Config:
    return Config(vocab_size=64, n_ctx=56, d_model=64, n_layers=2, n_heads=8, d_head=16, n_tasks=T.N_TAGS)


def summarize(effect: torch.Tensor) -> dict:
    mean, lo, hi = bootstrap_ci(effect.numpy())
    return {"mean": mean, "lo": lo, "hi": hi}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/q1"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-eval", type=int, default=2048)
    ap.add_argument("--n-ref", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--circuit-threshold", type=float, default=0.3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t_start = time.time()

    mixture = T.Mixture()
    induction, two_back = mixture.tasks
    (args.out / "models").mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "models" / f"seed{args.seed}.pt"
    train_log: list[dict] = []
    if ckpt.exists():
        model = Transformer.load(ckpt)
        print(f"[seed {args.seed}] loaded {ckpt}")
    else:
        model, train_log = train(model_config(), mixture, TrainConfig(steps=args.steps, seed=args.seed), verbose=False)
        model.save(ckpt)
        print(f"[seed {args.seed}] trained in {time.time() - t_start:.0f}s")
    model.eval()

    # Independent data streams for evaluation, mean reference, resampling and optimal fitting.
    g = lambda k: torch.Generator().manual_seed(1_000_000 * (k + 1) + args.seed)
    eval_batch = induction.sample(args.n_eval, g(0))
    ref_batch = induction.sample(args.n_ref, g(1))
    resample_batch = induction.sample(args.n_eval, g(2))
    fit_gen = g(3)
    ctx = AblationContext(model, eval_batch, ref_batch, resample_batch, lambda n: induction.sample(n, fit_gen))

    with torch.no_grad():
        clean = {}
        for name, task in (("induction", induction), ("two_back", two_back)):
            b = task.sample(args.n_eval, g(4))
            logits = model(b.tokens)
            clean[name] = {"loss": T.loss(logits, b).item(), "acc": T.accuracy(logits, b)}

    scores = head_scores(model, eval_batch)
    circuit = sorted(reference_circuit(scores, args.circuit_threshold))
    all_heads = ctx.all_heads
    non_circuit = [h for h in all_heads if h not in circuit]

    # How big is each head's constant part vs its input-dependent part?
    with torch.no_grad():
        names = [f"blocks.{l}.attn.result" for l in range(model.cfg.n_layers)] + ["resid_final"]
        _, cache = model.run_with_cache(ref_batch.tokens, names=names)
    L, H = model.cfg.n_layers, model.cfg.n_heads
    mean_norm = torch.zeros(L, H)
    dev_norm = torch.zeros(L, H)
    for l in range(L):
        r = cache[f"blocks.{l}.attn.result"]  # [b, p, h, d]
        mu = r.mean(0)
        mean_norm[l] = mu[1:].norm(dim=-1).mean(0)  # skip the tag position
        dev_norm[l] = (r - mu)[:, 1:].norm(dim=-1).mean((0, 1))
    resid_norm = cache["resid_final"][:, 1:].norm(dim=-1).mean().item()

    single: dict[str, dict] = {m: {} for m in METHODS}
    for (l, h) in all_heads:
        for method in METHODS:
            single[method][f"{l}.{h}"] = summarize(ablation_effect(ctx, [(l, h)], method))
        print(f"[seed {args.seed}] head {l}.{h} " + " ".join(f"{m}={single[m][f'{l}.{h}']['mean']:.4f}" for m in METHODS))

    groups = {
        "circuit": circuit,
        "non_circuit": non_circuit,
        "prev_token_heads": [h for h in circuit if h[0] == 0],
        "induction_heads": [h for h in circuit if h[0] >= 1],
        "layer0": [h for h in all_heads if h[0] == 0],
        "layer1": [h for h in all_heads if h[0] == 1],
    }
    group_effects = {}
    for name, heads in groups.items():
        if not heads:
            continue
        group_effects[name] = {"heads": [list(h) for h in heads]}
        for method in METHODS:
            group_effects[name][method] = summarize(ablation_effect(ctx, heads, method))
        print(f"[seed {args.seed}] group {name} " + " ".join(f"{m}={group_effects[name][m]['mean']:.3f}" for m in METHODS))

    out = {
        "seed": args.seed,
        "config": model.cfg.to_dict(),
        "train_steps": args.steps,
        "train_log": train_log,
        "clean": clean,
        "chance_loss": induction.chance_loss,
        "clean_eval_loss": ctx.clean_loss.mean().item(),
        "circuit_threshold": args.circuit_threshold,
        "head_scores": scores.to_dict(),
        "reference_circuit": [list(h) for h in circuit],
        "head_stats": {"mean_norm": mean_norm.tolist(), "dev_norm": dev_norm.tolist(), "resid_norm": resid_norm},
        "single": single,
        "groups": group_effects,
        "runtime_s": round(time.time() - t_start, 1),
    }
    path = args.out / f"seed{args.seed}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"[seed {args.seed}] wrote {path} in {out['runtime_s']}s")


if __name__ == "__main__":
    main()
