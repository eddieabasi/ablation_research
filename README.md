# ablation_research

Questions I ask, and try to prove, about ablation. #aiNative

Interpretability claims like "head X matters for behaviour Y" usually rest on an ablation,
and there are several ways to ablate (zero, mean, resample, optimal constant) that often
disagree. This repo tests ablation methods in **toy transformers trained from scratch on
tasks with a known mechanism**, so each conclusion can be checked against a reference
that does not itself come from ablation.

See [`docs/SCOPE.md`](docs/SCOPE.md) for the testbed, design decisions and full question list.

## Questions

| # | Question | Status |
|---|---|---|
| Q1 | Do zero, mean, resample and optimal ablation agree on which heads matter, and which recovers the known circuit? | **answered** → [`results/q1/REPORT.md`](results/q1/REPORT.md) |
| Q2 | How much of a zero-ablation effect is the mean shift rather than information? | planned |
| Q3 | Self-repair: do other components compensate when a circuit head is ablated? | planned |
| Q4 | Redundancy: does single-component ablation miss components that only matter jointly? | planned (Q1 already shows the effect) |
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
tests/test_core.py
results/q1/                checkpoints, per-seed JSON, summary, figures, REPORT.md
docs/SCOPE.md
```

## Running

CPU only; no downloads beyond PyPI.

```bash
pip install -e '.[dev]'
pytest                          # ~10 s
bash experiments/run_q1.sh 5    # 5 seeds; ~25-45 min on 4 cores, ~2.5 GB RAM per seed
```

Checkpoints in `results/q1/models/` are reused if present, so re-running only redoes
the ablations.
