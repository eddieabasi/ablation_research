# ablation_research

Questions I ask, and try to prove, about ablation. #aiNative

Interpretability claims like "head X matters for behaviour Y" usually rest on an ablation,
and there are several ways to ablate (zero, mean, resample, optimal constant) that often
disagree. This repo tests ablation methods in **toy transformers trained from scratch on
tasks with a known mechanism**, so each conclusion can be checked against a reference
that does not itself come from ablation.

See [`docs/SCOPE.md`](docs/SCOPE.md) for the testbed, design decisions and full question list.

## Findings so far

**Q1** ([report](results/q1/REPORT.md), 5 seeds): mean, resample and optimal ablation agree on
which heads matter (Spearman ρ 0.87–0.95) and recover the known circuit (AUROC 0.90–0.98).
Zero ablation ranks heads *worse than chance* (AUROC 0.25): it rates idle heads with large
constant outputs above real induction heads, and how far it diverges from mean ablation
is predicted by how constant a head's output is (ρ = 0.90). Methods that agree on rankings
still disagree on effect sizes by ~40× (resample vs optimal, previous-token heads), and
no method flags any single induction head, because 8 of them back each other up.

**Q2** ([report](results/q2/REPORT.md), 5 seeds): zero ablation is *not* "lost information +
off-distribution mean shift". Removing a head's constant alone explains only 31% (idle heads)
or 4% (previous-token heads) of zero ablation's excess over mean ablation; the rest appears
only when constant and information are removed together (superadditive in 5/5 seeds).
The compounding is inside the network (the logit margins compound; loss curvature explains
only 13–19%). LayerNorm doesn't cause the excess: the first LayerNorm downstream of an
ablated head *cushions* it (freezing LayerNorm makes 95% of single-head ablations worse), and
the final LayerNorm amplifies when nearly all signal is gone, which alone explains why
mean-ablating all of layer 0 hurts more than zero-ablating it. Path patching confirms
previous-token information reaches induction heads only through their keys.

**Q4** ([report](results/q4/REPORT.md), models with 2, 4 and 8 heads per layer): single-head
ablation misses components that only matter jointly, and the miss grows with redundancy. It finds
100% of induction heads with 2 heads per layer, 12% with 4, and 0% with 8. In the 8-head models,
removing half the induction heads costs 1–2% of removing all of them, yet keeping only one still
leaves 41% of the damage. Pairs don't fix it: every backup pair is superadditive, but singles plus
all pairwise interactions recover only 2% of the joint effect with 8 heads. Serial pairs
(previous-token → induction) are subadditive in the logit margin (97–100%) but can look
superadditive in the loss, because the saturated loss is convex. Single ablation also misses
*hidden backups* outside the reference circuit: weak previous-token heads that do nothing alone
but matter (up to 37% of the circuit effect) once the main previous-token head is gone.

## Questions

| # | Question | Status |
|---|---|---|
| Q1 | Do zero, mean, resample and optimal ablation agree on which heads matter, and which recovers the known circuit? | **answered** → [`results/q1/REPORT.md`](results/q1/REPORT.md) |
| Q2 | How much of a zero-ablation effect is the mean shift rather than information? | **answered** → [`results/q2/REPORT.md`](results/q2/REPORT.md) |
| Q3 | Self-repair: do other components compensate when a circuit head is ablated? | planned |
| Q4 | Redundancy: does single-component ablation miss components that only matter jointly? | **answered** → [`results/q4/REPORT.md`](results/q4/REPORT.md) |
| Q5 | How well does attribution patching predict real ablation effects? | planned |
| Q6 | How sensitive is resample ablation to the choice of corrupted distribution? | planned |

## Layout

```
src/ablation/
  model.py     hookable 2-layer transformer (per-head outputs exposed as hook points)
  tasks.py     induction + two-back tasks, mixture, loss/accuracy
  train.py     training loop
  circuit.py   ablation-free reference circuit from attention patterns
  ablate.py    zero / mean / resample / optimal-constant head ablation
  stats.py     bootstrap CIs, Spearman, AUROC
experiments/
  q1_method_agreement.py   one seed of Q1 → results/q1/seed<k>.json
  q1_analyze.py            aggregate seeds → summary.md, summary.json, figures
  run_q1.sh                all seeds in parallel, then aggregate
  q2_mean_shift.py         one seed of Q2 (reuses Q1 checkpoints) → results/q2/seed<k>.json
  q2_analyze.py            aggregate Q2 seeds
  run_q2.sh                Q2 seeds one at a time, then aggregate
  q2_followup.py           Q2 follow-up: LayerNorm split and margin test
  q4_redundancy.py         one model of Q4 (--heads, --seed) → results/q4/h<H>_seed<k>.json
  q4_analyze.py            aggregate Q4 models
  run_q4.sh                2/4/8 heads per layer, then aggregate
tests/test_core.py
results/q1/ … results/q4/  checkpoints, per-model JSON, summary, figures, REPORT.md
docs/SCOPE.md
```

## Running

CPU only; no downloads beyond PyPI.

```bash
pip install -e '.[dev]'
pytest                          # ~10 s
bash experiments/run_q1.sh 5    # 5 seeds; ~25-45 min on 4 cores, ~2.5 GB RAM per seed
bash experiments/run_q2.sh 5    # needs Q1 checkpoints; ~10 min per seed
bash experiments/run_q4.sh 5    # needs Q1 checkpoints; trains 2- and 4-head models to criterion
```

Checkpoints in `results/q1/models/` and `results/q4/models/` are reused if present, so re-running only redoes
the ablations.
