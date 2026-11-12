"""Q3 follow-up: is the final LayerNorm's repair norm restoration?

For every induction head of every Q3 model, under mean and resample ablation: the norm of
the replacement output relative to the clean output, and the ratio of the final
LayerNorm's scale after vs before the ablation (scored positions). If LayerNorm repair
is norm restoration, mean ablation (small replacement) should shrink the residual stream
and the scale, and resample (normal-sized replacement) should not.

    python experiments/q3_norm_check.py results/q3

Writes ``results/q3/norm_check.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from ablation import tasks as T
from ablation.ablate import AblationContext, ablation_hooks, result_hook_name
from ablation.model import Transformer

N_EVAL = 1024


def model_path(r: dict) -> Path:
    L, H, s = r["layers"], r["heads"], r["seed"]
    if L == 2 and H == 8:
        return Path("results/q1/models") / f"seed{s}.pt"
    if L == 2:
        return Path("results/q4/models") / f"h{H}_seed{s}.pt"
    return Path("results/q3/models") / f"L{L}_h{H}_seed{s}.pt"


@torch.no_grad()
def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/q3")
    torch.set_num_threads(4)
    out = []
    for p in sorted(d.glob("L*_h*_seed*.json")):
        r = json.loads(p.read_text())
        model = Transformer.load(model_path(r))
        model.eval()
        induction = T.Mixture().tasks[0]
        g = lambda k: torch.Generator().manual_seed(1_000_000 * (k + 1) + r["seed"])  # same streams as Q3
        ctx = AblationContext(model, induction.sample(N_EVAL, g(0)), induction.sample(4096, g(1)),
                              induction.sample(N_EVAL, g(2)), lambda n: induction.sample(n, g(3)))
        batch = ctx.eval_batch
        w = batch.mask.float()
        smean = lambda x: float(((x * w).sum() / w.sum()))
        s_clean = ctx.clean_ln_scales["ln_final.scale"].squeeze(-1)
        for k, role in r["roles"].items():
            if role != "ind":
                continue
            l, h = map(int, k.split("."))
            clean_out = ctx.clean_result[l][:, :, h]
            row = {"model": p.stem, "layers": r["layers"], "heads": r["heads"], "seed": r["seed"], "head": k,
                   "clean_out_norm": smean(clean_out.norm(dim=-1))}
            for method in ("mean", "resample"):
                hooks = ablation_hooks(ctx, [(l, h)], method)
                repl = hooks[result_hook_name(l)](ctx.clean_result[l])[:, :, h]
                _, c = model.run_with_cache(batch.tokens, hooks=hooks, names=["ln_final.scale"])
                row[method] = {
                    "repl_norm_ratio": smean(repl.norm(dim=-1)) / row["clean_out_norm"],
                    "scale_ratio": smean(c["ln_final.scale"].squeeze(-1) / s_clean),
                }
            out.append(row)
        print(p.stem, "done", flush=True)
    (d / "norm_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
