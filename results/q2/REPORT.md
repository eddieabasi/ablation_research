# Q2: How much of a zero-ablation effect is the mean shift rather than information?

**Question.** Q1 found that zero ablation misranks heads, and that its divergence from mean
ablation tracks how constant a head's output is. The natural reading is that zero ablation
= (losing the information) + (losing the constant, which pushes activations off-distribution).
How much of a zero-ablation effect is each, and what carries the damage?

**Short answer.** The question assumes the two parts add up, and they don't.

* **Removing the constant alone does little.** For the idle non-circuit heads it accounts
  for only 31% of zero ablation's excess over mean ablation; for previous-token heads, 4%.
  Most of the excess appears only when the constant and the information are removed
  *together* (positive interaction in 5/5 seeds; 39 of 40 single layer-0 heads).
* **The compounding happens inside the network, not just in the loss.** The target-logit
  margins themselves compound (31 of 31 idle heads, 9 of 9 previous-token heads). The
  curvature of the loss explains only 13–19% of the loss interaction.
* **LayerNorm is not the culprit.** It cushions single-head ablations: freezing it at its
  clean scale makes 95% of zero ablations and 100% of mean ablations *worse*. The layer-1
  LayerNorm does the cushioning for layer-0 heads, the final LayerNorm for layer-1 heads.
  The final LayerNorm also *amplifies* when an ablation removes almost all the useful
  signal, and that alone explains the Q1 puzzle of mean ablation beating zero on all of
  layer 0.
* **The damage has specific routes.** The idle heads' constant acts through layer-1
  queries (with keys). The previous-token heads' information acts entirely through layer-1
  keys, the textbook induction mechanism, confirmed causally here.

## Design

Same 5 models, data streams and reference circuit as Q1 ([`docs/SCOPE.md`](../../docs/SCOPE.md)).
Write a head's output as r = μ + ε: the per-position mean μ plus the input-dependent
deviation ε. A 2×2 design:

| ablation | replaces r with | removes |
|---|---|---|
| mean | μ | the information ε |
| shift | ε | the constant μ |
| zero | 0 | both |

Interaction = zero − mean − shift. The Shapley share of the constant (its average marginal
contribution over the two orders) is ½[shift + (zero − mean)] / zero.

Each ablation is also run (a) with **every LayerNorm frozen** at its clean-run scale, so the
ablation can change the residual stream's direction but not its norm, and separately with
only the layer-1 or only the final LayerNorm frozen, and (b) for layer-0
heads, **path-specifically**: the change reaches only the layer-1 queries, keys, values, or
the unembedding directly. In a 2-layer model, all four paths together reproduce the full
ablation exactly (tested in `tests/test_core.py`).

A follow-up (`experiments/q2_followup.py`) adds the single-LayerNorm freezes and a
**margin test**. Loss is exactly softplus(−m) of the target-logit margin
m = z_target − logsumexp(other logits), a fixed convex function. If the margin drops of
mean and shift added up to zero's, the loss interaction would come purely from that
curvature.

The zero and mean numbers reproduce Q1 exactly (same evaluation sequences). Full tables:
[`summary.md`](summary.md); per-seed data in `seed*.json` and `followup_seed*.json`.

## Findings

### 1. Zero ≠ information + constant: the two interact

![decomposition](decomposition.png)

Effects in nats (mean ± sd over 5 seeds):

| ablated together | zero | mean (info) | shift (constant) | interaction | constant's Shapley share |
|---|---|---|---|---|---|
| all non-circuit heads | 0.71 ± 0.35 | 0.026 ± 0.029 | 0.21 ± 0.07 | **+0.47 ± 0.30** | 66% |
| previous-token heads | 3.58 ± 2.30 | 2.74 ± 2.31 | 0.031 ± 0.027 | **+0.81 ± 0.32** | 17% |
| induction heads | 5.34 ± 0.06 | 5.34 ± 0.05 | 0.000 | 0.00 ± 0.04 | 0% |
| all of layer 0 | 6.19 ± 0.21 | 6.66 ± 0.14 | 0.58 ± 0.28 | **−1.04 ± 0.26** | 1% |

Single heads (median over heads pooled across seeds): constant's Shapley share is **83%**
for idle non-circuit heads (n = 31), **34%** for previous-token heads (n = 9), and **2%**
for induction heads (n = 40).

So "how much is the mean shift" has an order-dependent answer:

* **Idle heads.** Removing the constant alone costs 0.21 nats, and removing their
  information alone costs 0.03. Removing both costs 0.71, three times the sum. The idle
  heads' small input-dependent signal (Q1: weak previous-token attention) is harmless to
  lose while the constant is in place, and costly once it isn't.
* **Previous-token heads.** Their constant alone is almost irrelevant (0.03 nats). But
  zero ablation costs 0.84 nats more than mean ablation, and that excess is nearly all
  interaction. With the information gone, the constant still matters.
* **Induction heads.** Their outputs are almost entirely input-dependent, so zero and
  mean ablation agree (as Q1 found) and there is nothing to decompose.
* **All of layer 0.** The interaction turns *negative*: the loss is already near its
  ceiling, so the second removal adds less.

**Is it just the loss curvature? Mostly not.** An earlier version of this report proposed
that the superadditivity came from saturation: the clean model is very confident (median
target-logit margin 7.5 nats, i.e. about 99.94% on the right token), and loss is steeply
convex in the margin. The margin test checks this directly (mean ± sd over seeds;
single heads: median):

| ablated | margin drop: mean (info) | shift (constant) | zero (both) | margin interaction | share of loss interaction explained by curvature |
|---|---|---|---|---|---|
| all idle heads | 0.42 | 3.32 | 4.89 | **+1.15 ± 0.51** (5/5 seeds > 0) | 19% ± 7% |
| previous-token heads | 7.60 | 1.05 | 9.17 | +0.53 ± 1.11 (3/5 seeds > 0) | varies by seed |
| single idle heads (n = 31) | 0.03 | 0.40 | 0.44 | +0.02 (31/31 > 0) | 13% |
| single previous-token heads (n = 9) | 1.33 | 0.48 | 2.60 | **+0.77** (9/9 > 0) | 16% |

Saturation is real: removing the idle heads' constant cuts the margin by 3.3 of its 7.5
nats yet costs only 0.21 nats of loss, so loss hides large internal effects. But the
superadditivity is mostly *inside* the network: the margins themselves compound, and
curvature would produce only about a fifth of the loss interaction if they didn't. The
exception proves the pattern. In the two seeds where ablating the previous-token heads
already destroys induction (about 6 nats), the margin interaction turns negative and the
remaining loss interaction is curvature, just as for all of layer 0, where margins are
strongly subadditive (−4.9). Where in the network the margins compound (the attention
softmax, the copy readout, LayerNorm) is not localised further here.

### 2. LayerNorm cushions ablations; it amplifies only when the signal is gone

![layernorm](layernorm.png)

Effect with LayerNorm frozen ÷ effect with it live (single heads with live effect > 0.001
nats, median [IQR]):

| head type | zero | mean | shift |
|---|---|---|---|
| previous-token (n = 9) | 2.0 [1.6, 2.5] | 4.1 [3.0, 5.6] | 1.9 [1.6, 2.2] |
| induction (n = 26) | 3.4 [3.0, 3.7] | 3.4 [3.0, 3.8] | – |
| idle non-circuit (n = 31; shift n = 30) | 1.6 [1.1, 2.1] | – | 1.2 [1.0, 1.5] |

Freezing LayerNorm makes **95%** of single-head zero ablations, **100%** of mean ablations
and **79%** of shift ablations worse. The original hypothesis (LayerNorm rescaling creates
zero ablation's excess) is refuted. LayerNorm sees only the residual stream's direction,
so any part of an ablation that merely grows or shrinks the residual is cancelled, and
what remains (for an induction head, the seven others doing the same job) is rescaled to
full strength. That compensation is a form of self-repair; Rushing & Nanda (2024)
attribute part of self-repair in real LMs to LayerNorm. Q3 will measure this directly.

**Which LayerNorm?** Freezing one at a time (ratio to live; single heads: median; groups:
mean over seeds):

| ablated | method | freeze layer-1 LayerNorm only | freeze final LayerNorm only | freeze both |
|---|---|---|---|---|
| single previous-token head (n = 9) | zero | **2.23** | 0.96 | 2.04 |
| | mean | **4.06** | 1.00 | 4.07 |
| single induction head (n = 26) | zero | 1.00 | **3.36** | 3.36 |
| | mean | 1.00 | **3.40** | 3.40 |
| single idle head (n = 31) | zero | 1.26 | 1.20 | 1.63 |
| all induction heads | mean | 1.00 | **0.79** | 0.79 |
| all of layer 0 | mean | 1.06 | **0.78** | 0.74 |

The cushioning happens at the first LayerNorm downstream of the ablated head: layer 1's
for layer-0 heads, the final one for layer-1 heads. (Freezing layer 1's LayerNorm cannot
affect a layer-1 head's ablation, hence the exact 1.00s.) The amplification described
next is the final LayerNorm's.

The same rescaling **backfires when an ablation removes almost all the useful signal**.
Ablating all induction heads at once shrinks the final residual to 0.30× its clean scale;
LayerNorm scales the remains back up, and the model is confidently wrong. With LayerNorm
frozen, it falls back towards uniform (output entropy 4.0 nats) and the loss is 21%
lower (ratio 0.79).

**This resolves the Q1 layer-0 puzzle.** Mean-ablating all of layer 0 removes every head's
deviation ε, which shrinks the layer-1 residual to 0.86× and the final residual to 0.69×
of clean scale. Zero ablation leaves them at 1.04× and 0.89×. LayerNorm therefore
amplifies the now-uninformative signal more under mean ablation. The model becomes more
confident (output entropy 1.07 vs 1.75 nats) while equally wrong: answer-key attention is
0.06 under both, against 0.88 clean. Freeze LayerNorm and the order flips in **5/5 seeds**:
mean 4.93 vs zero 5.90 nats on average, against 6.66 vs 6.19 live. Freezing **only the
final LayerNorm** is enough: mean 5.23 vs zero 5.83, again 5/5 seeds.

### 3. Where the damage travels

![paths](paths.png)

Path effect as a fraction of the full effect (mean ± sd over seeds):

| ablated | method | q only | k only | v only | direct | q + k |
|---|---|---|---|---|---|---|
| idle heads | shift (constant) | **0.39 ± 0.20** | 0.08 ± 0.03 | 0.04 | 0.00 | 0.75 ± 0.15 |
| idle heads | zero | 0.31 ± 0.27 | 0.14 ± 0.11 | 0.01 | 0.00 | 0.89 ± 0.08 |
| previous-token heads | mean (info) | 0.00 | **1.12 ± 0.09** | 0.00 | 0.00 | 1.10 ± 0.06 |
| previous-token heads | zero | 0.00 | **1.01 ± 0.05** | 0.00 | 0.00 | 1.05 ± 0.04 |

* **Previous-token information acts only through layer-1 keys.** That is the textbook
  K-composition: the induction head's key at position s must say "the previous token was
  X". Here it is confirmed causally, not inferred from weights. (The keys-only effect
  slightly exceeds the full effect: the accompanying change to the queries offsets a
  little of the damage.)
* **The idle heads' constant acts mainly through layer-1 queries**, and fully only when
  queries and keys are perturbed together (q alone 0.39 and k alone 0.08 sum to 0.47,
  against 0.75 jointly). Superadditivity again. The values and the direct path to the
  unembedding play no part.

### 4. What predicts the constant's damage

Spearman ρ, pooled over seeds:

| heads | target | size of mean output | how far it moves layer-1 **queries** | … **keys** | … **values** |
|---|---|---|---|---|---|
| idle (n = 31) | shift effect | 0.60 | **0.74** | 0.60 | 0.49 |
| all layer 0 (n = 40) | zero − mean | 0.76 | 0.50 | **0.81** | 0.51 |

A head's mean output predicts its damage better when measured as *how far it moves the
inputs that matter*. For idle heads that is the layer-1 queries; for the full set including
previous-token heads it is the keys, consistent with the path results. Both "moves"
estimates are first-order (through the clean LayerNorm scale), so this is a cheap
screening statistic, not a substitute for running the ablation.

## Verdicts on the Q2 hypotheses

| hypothesis (from `SCOPE.md`) | verdict |
|---|---|
| zero − mean is the cost of the mean shift | **Mostly no.** The shift alone is 31% of zero − mean for idle heads and 4% for previous-token heads; the rest is interaction. |
| it is predicted by the norm of the head's mean output | **Partly** (ρ = 0.60–0.76). How far the mean moves the relevant layer-1 reads predicts as well or better. |
| it is mediated by LayerNorm rescaling | **No.** LayerNorm cushions single-head ablations; it amplifies only when nearly all signal is removed. |

## What this means in practice

1. **Don't read zero − mean as "the cost of the off-distribution shift".** Run the shift
   ablation (r → r − μ) directly. It is one extra forward pass.
2. **Expect ablation effects to be non-additive.** Here the compounding is mostly inside
   the network (in the logit margins), with saturated loss adding a little on top.
   Decompositions of the form "X% due to A, Y% due to B" depend on order. Report both
   orders, or a Shapley split, and say which. Checking interactions in margin space as
   well as loss space shows how much the loss itself is responsible.
3. **Freezing LayerNorm is not a neutral "cleaner" ablation.** It switches off a
   compensation mechanism: single-head effects grow 1.2–4×, while large group ablations
   shrink. State which you use.
4. **Path-specific ablation is a cheap causal check of a hypothesised mechanism.** Here it
   confirmed K-composition in one line of output.

## Caveats

* Same toy setting as Q1: 2-layer attention-only models on synthetic tasks, clean loss
  about 0.001. That saturation hides large margin effects from the loss; real LMs are far
  less saturated, so loss-level effect sizes will differ. The margin-level interactions do
  not depend on saturation, but have only been measured in this setting.
* The Shapley split has only two players and is one convention among several; the raw
  2×2 numbers are in [`summary.md`](summary.md).
* In a 2-layer model, path effects are exact; in deeper models they measure direct paths
  only.
* The idle heads' LayerNorm ratio under mean ablation rests on one head; mean ablation of
  the others is too small (< 0.001 nats) to give a meaningful ratio.

## Reproduce

```bash
pip install -e '.[dev]'
bash experiments/run_q2.sh 5     # needs the Q1 checkpoints in results/q1/models/
for s in 0 1 2 3 4; do python experiments/q2_followup.py --seed $s; done   # LayerNorm split, margin test
python experiments/q2_analyze.py results/q2
```

About 9–14 minutes per seed with 2–4 threads for the main run, 6–9 for the follow-up,
and about 2.7 GB of memory per seed.
