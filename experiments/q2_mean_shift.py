"""Q2: How much of a zero-ablation effect is the mean shift rather than information?

A head's output is r = mu + eps (per-position mean + input-dependent deviation). Three
ablations form a 2x2 design:

    mean   r -> mu    removes the information eps, keeps the constant
    shift  r -> eps   removes the constant mu, keeps the information
    zero   r -> 0     removes both

so zero = mean + shift + interaction. Each is run with LayerNorm live and frozen at its
clean scale, and (for layer-0 heads) path-specifically: the change delivered only to the
layer-1 queries, keys, values, or straight to the unembedding.

Reuses the Q1 checkpoints and the same data streams as Q1. For one seed:

    python experiments/q2_mean_shift.py --seed 0

Writes ``results/q2/seed<seed>.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ablation import tasks as T
from ablation.ablate import RECEIVERS, AblationContext, effect_hooks
from ablation.model import Transformer
from ablation.stats import bootstrap_ci

METHODS = ("zero", "mean", "shift")


def answer_mask(tokens: torch.Tensor) -> torch.Tensor:
    """[b, t, s] True where key s holds the induction answer for query t (tokens[s-1] == tokens[t])."""
    b, n = tokens.shape
    prev_of_key = torch.cat([torch.full((b, 1), -1), tokens[:, :-1]], dim=1)
    ans = prev_of_key.unsqueeze(1) == tokens.unsqueeze(2)
    return ans & torch.ones(n, n, dtype=torch.bool).tril(0)


class Runner:
    def __init__(self, ctx: AblationContext):
        self.ctx = ctx
        self.model = ctx.model
        self.batch = ctx.eval_batch
        self.ans = answer_mask(self.batch.tokens)
        self.mask = self.batch.mask
        self.last = self.model.cfg.n_layers - 1
        self.diag_names = [f"blocks.{self.last}.attn.pattern", f"blocks.{self.last}.ln1.q.scale", "ln_final.scale"]
        self.clean = self._run({})

    def _masked(self, x: torch.Tensor) -> float:
        """mean of x [b, t] over induction positions"""
        m = self.mask.float()
        return float((x * m).sum() / m.sum())

    @torch.no_grad()
    def _run(self, hooks) -> dict:
        logits, cache = self.model.run_with_cache(self.batch.tokens, hooks=hooks, names=self.diag_names)
        pat = cache[self.diag_names[0]]  # [b, h, t, s]
        ind_mass = (pat * self.ans.unsqueeze(1)).sum(-1).mean(1)  # [b, t], averaged over heads
        lp = logits.log_softmax(-1)
        out_ent = -(lp.exp() * lp).sum(-1)
        loss = T.per_seq_loss(logits, self.batch)
        return {
            "per_seq_loss": loss,
            "induction_attention": self._masked(ind_mass),
            "output_entropy": self._masked(out_ent),
            "accuracy": T.accuracy(logits, self.batch),
            "ln_q_scale": cache[self.diag_names[1]].squeeze(-1),
            "ln_final_scale": cache[self.diag_names[2]].squeeze(-1),
        }

    def measure(self, heads, method, freeze_ln=False, receivers=None, diagnostics=True) -> dict:
        hooks = effect_hooks(self.ctx, heads, method, freeze_ln=freeze_ln, receivers=receivers)
        if not diagnostics:
            with torch.no_grad():
                logits = self.model(self.batch.tokens, hooks=hooks)
            eff = T.per_seq_loss(logits, self.batch) - self.ctx.clean_loss
            m, lo, hi = bootstrap_ci(eff.numpy())
            return {"effect": m, "lo": lo, "hi": hi}
        d = self._run(hooks)
        eff = d["per_seq_loss"] - self.ctx.clean_loss
        m, lo, hi = bootstrap_ci(eff.numpy())
        return {
            "effect": m,
            "lo": lo,
            "hi": hi,
            "induction_attention": d["induction_attention"],
            "output_entropy": d["output_entropy"],
            "accuracy": d["accuracy"],
            "ln_q_scale_ratio": self._masked(d["ln_q_scale"] / self.clean["ln_q_scale"]),
            "ln_final_scale_ratio": self._masked(d["ln_final_scale"] / self.clean["ln_final_scale"]),
        }


def static_reads(model: Transformer, ctx: AblationContext) -> dict:
    """How strongly each layer-0 head's mean output moves later layers' q/k/v inputs.

    First-order estimate: the mean output mu_h, centred and divided by the clean layer-1
    LayerNorm scale, times the LayerNorm gain, projected through each layer-1 W_Q/W_K/W_V
    (root-sum-square over layer-1 heads, mean over non-tag positions).
    """
    mu = ctx.mean_result[0]  # [pos, head, d]
    scales = ctx.clean_ln_scales
    out = {"mean_norm": mu[1:].norm(dim=-1).mean(0).tolist()}
    blk = model.blocks[1]
    for c, W in (("q", blk.attn.W_Q), ("k", blk.attn.W_K), ("v", blk.attn.W_V)):
        sigma = scales[f"blocks.1.ln1.{c}.scale"].mean(0)  # [pos, 1]
        x = (mu - mu.mean(-1, keepdim=True)) / sigma.unsqueeze(1) * blk.ln1.weight
        read = torch.einsum("phd,gde->phge", x, W).norm(dim=-1)  # [pos, head, l1_head]
        out[f"{c}_read"] = read.pow(2).sum(-1).sqrt()[1:].mean(0).tolist()
    resid = ctx.model.run_with_cache(ctx.eval_batch.tokens, names=["blocks.1.resid_pre"])[1]["blocks.1.resid_pre"]
    out["resid_norm"] = float(resid[:, 1:].norm(dim=-1).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--q1", type=Path, default=Path("results/q1"))
    ap.add_argument("--out", type=Path, default=Path("results/q2"))
    ap.add_argument("--n-eval", type=int, default=2048)
    ap.add_argument("--n-ref", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    model = Transformer.load(args.q1 / "models" / f"seed{args.seed}.pt")
    q1 = json.loads((args.q1 / f"seed{args.seed}.json").read_text())
    task = T.InductionTask()
    # Same data streams as Q1 (see q1_method_agreement.py), so effects are comparable.
    g = lambda k: torch.Generator().manual_seed(1_000_000 * (k + 1) + args.seed)
    fit_gen = g(3)
    ctx = AblationContext(
        model, task.sample(args.n_eval, g(0)), task.sample(args.n_ref, g(1)), task.sample(args.n_eval, g(2)),
        lambda n: task.sample(n, fit_gen),
    )
    run = Runner(ctx)

    circuit = [tuple(h) for h in q1["reference_circuit"]]
    all_heads = ctx.all_heads
    groups = {
        "non_circuit": [h for h in all_heads if h not in circuit],
        "prev_token_heads": [h for h in circuit if h[0] == 0],
        "induction_heads": [h for h in circuit if h[0] >= 1],
        "layer0": [h for h in all_heads if h[0] == 0],
    }
    units = {f"{l}.{h}": [(l, h)] for l, h in all_heads} | {f"group:{k}": v for k, v in groups.items() if v}

    results: dict = {}
    for name, heads in units.items():
        entry: dict = {"heads": [list(h) for h in heads]}
        for method in METHODS:
            entry[method] = run.measure(heads, method)
            entry[f"{method}_frozen_ln"] = run.measure(heads, method, freeze_ln=True)
        if all(l == 0 for l, _ in heads):
            for method in METHODS:
                entry[f"{method}_paths"] = {
                    r: run.measure(heads, method, receivers=[r], diagnostics=False)["effect"] for r in RECEIVERS
                }
                entry[f"{method}_paths"]["qk"] = run.measure(heads, method, receivers=["q", "k"], diagnostics=False)["effect"]
        results[name] = entry
        print(f"[seed {args.seed}] {name:24s} " + " ".join(
            f"{m}={entry[m]['effect']:.4f}/{entry[m + '_frozen_ln']['effect']:.4f}" for m in METHODS), flush=True)

    clean = {k: v for k, v in run.clean.items() if not isinstance(v, torch.Tensor)}
    out = {
        "seed": args.seed,
        "reference_circuit": [list(h) for h in circuit],
        "head_scores": q1["head_scores"],
        "clean": clean,
        "clean_eval_loss": ctx.clean_loss.mean().item(),
        "chance_loss": task.chance_loss,
        "static_reads_layer0": static_reads(model, ctx),
        "units": results,
        "runtime_s": round(time.time() - t0, 1),
    }
    path = args.out / f"seed{args.seed}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"[seed {args.seed}] wrote {path} in {out['runtime_s']}s", flush=True)


if __name__ == "__main__":
    main()
