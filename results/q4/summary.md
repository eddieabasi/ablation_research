# Q4 summary

Effects are increases in induction loss (nats) on the Q1 evaluation sets. Primary method: mean ablation. Interaction of a pair = effect(both) − effect(A) − effect(B); significant = 95% bootstrap CI over sequences excludes 0. Margin interaction: same, on the target-logit margin (> 0 = superadditive inside the network). Explained = share of the loss interaction that loss curvature alone would produce if margin drops added.

## Models

| heads/layer | seeds | training steps | induction acc | two-back acc | prev-token heads | induction heads |
|---|---|---|---|---|---|---|
| 2 | 6 (0, 14, 16, 17, 27, 29) | 1500, 1500, 1500, 1500, 1500, 1500 | 0.9980 ± 0.0020 | 1.0000 ± 0.0000 | 1.2 ± 0.4 | 2.0 ± 0.0 |
| 4 | 4 (1, 2, 3, 4) | 1500, 1500, 1500, 1500 | 0.9999 ± 0.0001 | 0.9999 ± 0.0000 | 1.2 ± 0.5 | 4.0 ± 0.0 |
| 8 | 5 (0, 1, 2, 3, 4) | 1500, 1500, 1500, 1500, 1500 | 0.9999 ± 0.0000 | 1.0000 ± 0.0000 | 1.8 ± 0.8 | 8.0 ± 0.0 |

Excluded models:

* 2 heads/layer, seed 1: missed 99% accuracy after 10000 steps (induction accuracy 0.794)
* 2 heads/layer, seed 10: missed 99% accuracy after 1500 steps (induction accuracy 0.195)
* 2 heads/layer, seed 11: missed 99% accuracy after 1500 steps (induction accuracy 0.238)
* 2 heads/layer, seed 12: missed 99% accuracy after 1500 steps (induction accuracy 0.179)
* 2 heads/layer, seed 13: missed 99% accuracy after 1500 steps (induction accuracy 0.171)
* 2 heads/layer, seed 15: missed 99% accuracy after 1500 steps (induction accuracy 0.215)
* 2 heads/layer, seed 18: missed 99% accuracy after 1500 steps (induction accuracy 0.215)
* 2 heads/layer, seed 19: missed 99% accuracy after 1500 steps (induction accuracy 0.175)
* 2 heads/layer, seed 2: missed 99% accuracy after 10000 steps (induction accuracy 0.686)
* 2 heads/layer, seed 20: missed 99% accuracy after 1500 steps (induction accuracy 0.198)
* 2 heads/layer, seed 21: missed 99% accuracy after 1500 steps (induction accuracy 0.166)
* 2 heads/layer, seed 22: missed 99% accuracy after 1500 steps (induction accuracy 0.186)
* 2 heads/layer, seed 23: missed 99% accuracy after 1500 steps (induction accuracy 0.210)
* 2 heads/layer, seed 24: missed 99% accuracy after 1500 steps (induction accuracy 0.186)
* 2 heads/layer, seed 25: missed 99% accuracy after 1500 steps (induction accuracy 0.174)
* 2 heads/layer, seed 26: missed 99% accuracy after 1500 steps (induction accuracy 0.198)
* 2 heads/layer, seed 28: missed 99% accuracy after 1500 steps (induction accuracy 0.241)
* 2 heads/layer, seed 3: missed 99% accuracy after 10000 steps (induction accuracy 0.363)
* 2 heads/layer, seed 30: missed 99% accuracy after 1500 steps (induction accuracy 0.194)
* 2 heads/layer, seed 31: missed 99% accuracy after 1500 steps (induction accuracy 0.203)
* 2 heads/layer, seed 32: missed 99% accuracy after 1500 steps (induction accuracy 0.188)
* 2 heads/layer, seed 33: missed 99% accuracy after 1500 steps (induction accuracy 0.180)
* 2 heads/layer, seed 34: missed 99% accuracy after 1500 steps (induction accuracy 0.173)
* 2 heads/layer, seed 4: missed 99% accuracy after 10000 steps (induction accuracy 0.747)
* 2 heads/layer, seed 5: missed 99% accuracy after 1500 steps (induction accuracy 0.189)
* 2 heads/layer, seed 6: missed 99% accuracy after 1500 steps (induction accuracy 0.208)
* 2 heads/layer, seed 7: missed 99% accuracy after 1500 steps (induction accuracy 0.185)
* 2 heads/layer, seed 8: missed 99% accuracy after 1500 steps (induction accuracy 0.169)
* 2 heads/layer, seed 9: missed 99% accuracy after 1500 steps (induction accuracy 0.345)
* 4 heads/layer, seed 0: missed 99% accuracy after 10000 steps (induction accuracy 0.691)

## 1. Does single-head ablation find the circuit heads?

Recall = fraction of circuit heads whose single-head effect exceeds 1% of the whole-circuit effect. Single ÷ group = one head's effect ÷ effect of ablating all heads of its kind (median over heads). Σ singles ÷ group = 1 if effects add up, ≪ 1 under redundancy. + pairs = the same with every pairwise interaction within the group added (second-order inclusion–exclusion).

| heads/layer | method | recall: prev-token | recall: induction | single ÷ group: induction | Σ singles ÷ group: induction | + pairs ÷ group: induction | Σ singles ÷ group: prev-token |
|---|---|---|---|---|---|---|---|
| 2 | mean | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.060 | 0.122 ± 0.035 | 1.000 ± 0.000 | 0.858 ± 0.347 |
| 2 | resample | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.150 | 0.335 ± 0.025 | 1.000 ± 0.000 | 0.946 ± 0.132 |
| 4 | mean | 1.00 ± 0.00 | 0.12 ± 0.25 | 0.001 | 0.022 ± 0.030 | 0.206 ± 0.184 | 0.778 ± 0.443 |
| 4 | resample | 1.00 ± 0.00 | 0.31 ± 0.24 | 0.006 | 0.065 ± 0.075 | 0.811 ± 0.112 | 0.868 ± 0.263 |
| 8 | mean | 0.83 ± 0.24 | 0.00 ± 0.00 | 0.000 | 0.002 ± 0.001 | 0.020 ± 0.006 | 0.453 ± 0.501 |
| 8 | resample | 0.93 ± 0.15 | 0.00 ± 0.00 | 0.000 | 0.004 ± 0.002 | 0.129 ± 0.043 | 0.638 ± 0.349 |

## 2. Pair interactions by pair type

| heads/layer | method | pair type | n | median interaction | median relative (÷ effect of both) | sig. > 0 | sig. < 0 | median margin interaction | margin < 0 | median explained by curvature |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | mean | ind-ind | 6 | 4.7579 | 0.87 | 100% | 0% | 3.767 | 0% | 0.29 |
| 2 | mean | prev-prev | 1 | 6.5051 | 0.85 | 100% | 0% | 9.737 | 0% | 0.05 |
| 2 | mean | prev-ind | 14 | -0.2766 | -0.04 | 43% | 57% | -3.591 | 100% | -2.92 |
| 2 | mean | circuit-non | 15 | 0.0090 | 0.03 | 87% | 7% | 0.021 | 27% | 0.33 |
| 2 | resample | ind-ind | 6 | 8.4895 | 0.66 | 100% | 0% | 3.845 | 0% | 0.54 |
| 2 | resample | prev-prev | 1 | 2.9128 | 0.32 | 100% | 0% | 1.253 | 0% | 0.87 |
| 2 | resample | prev-ind | 14 | -1.6966 | -0.21 | 21% | 79% | -6.691 | 100% | -2.51 |
| 2 | resample | circuit-non | 15 | 0.1147 | 0.05 | 80% | 13% | 0.026 | 40% | 0.75 |
| 4 | mean | ind-ind | 24 | 0.0977 | 0.81 | 100% | 0% | 0.590 | 0% | 0.24 |
| 4 | mean | prev-prev | 1 | 2.9505 | 0.89 | 100% | 0% | 6.159 | 0% | 0.04 |
| 4 | mean | prev-ind | 20 | 0.0588 | 0.16 | 80% | 20% | -0.309 | 100% | 1.37 |
| 4 | mean | circuit-non | 57 | 0.0013 | 0.09 | 79% | 21% | 0.001 | 46% | 0.76 |
| 4 | mean | non-non | 10 | 0.0014 | 0.38 | 90% | 0% | 0.012 | 10% | 0.60 |
| 4 | resample | ind-ind | 24 | 1.2258 | 0.89 | 100% | 0% | 1.728 | 0% | 0.31 |
| 4 | resample | prev-prev | 1 | 3.9405 | 0.53 | 100% | 0% | 3.891 | 0% | 0.37 |
| 4 | resample | prev-ind | 20 | -0.1243 | -0.03 | 40% | 60% | -1.488 | 100% | -2.50 |
| 4 | resample | circuit-non | 57 | 0.0302 | 0.18 | 95% | 0% | 0.017 | 39% | 0.78 |
| 4 | resample | non-non | 10 | 0.0021 | 0.39 | 100% | 0% | 0.029 | 0% | 0.27 |
| 8 | mean | ind-ind | 140 | 0.0024 | 0.44 | 100% | 0% | 0.110 | 0% | 0.33 |
| 8 | mean | prev-prev | 5 | 1.3442 | 0.88 | 100% | 0% | 3.766 | 0% | 0.05 |
| 8 | mean | prev-ind | 72 | 0.0210 | 0.14 | 100% | 0% | -0.064 | 97% | 1.08 |
| 8 | mean | circuit-non | 301 | 0.0003 | 0.10 | 70% | 0% | -0.002 | 57% | 0.66 |
| 8 | mean | non-non | 82 | 0.0002 | 0.27 | 61% | 0% | 0.003 | 32% | 0.27 |
| 8 | resample | ind-ind | 140 | 0.0329 | 0.75 | 100% | 0% | 0.451 | 0% | 0.30 |
| 8 | resample | prev-prev | 5 | 3.4164 | 0.68 | 100% | 0% | 4.124 | 0% | 0.25 |
| 8 | resample | prev-ind | 72 | 0.0177 | 0.02 | 62% | 36% | -0.326 | 100% | 2.46 |
| 8 | resample | circuit-non | 301 | 0.0019 | 0.20 | 99% | 0% | 0.003 | 46% | 0.92 |
| 8 | resample | non-non | 82 | 0.0014 | 0.41 | 99% | 0% | 0.019 | 7% | 0.27 |

## 3. Nested ablation of induction heads (mean ablation)

Effect of removing k of n induction heads (mean over random subsets) ÷ effect of removing all n.

**2 heads/layer** (removable before the effect passes 10% of full: 0.50 ± 0.00 of the induction heads)

| seed | n | k=1 | k=2 |
|---|---|---|---|
| 0 | 2 | 0.034 | 1.000 |
| 14 | 2 | 0.069 | 1.000 |
| 16 | 2 | 0.049 | 1.000 |
| 17 | 2 | 0.084 | 1.000 |
| 27 | 2 | 0.071 | 1.000 |
| 29 | 2 | 0.059 | 1.000 |

**4 heads/layer** (removable before the effect passes 10% of full: 0.44 ± 0.12 of the induction heads)

| seed | n | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|
| 1 | 4 | 0.017 | 0.103 | 0.367 | 1.000 |
| 2 | 4 | 0.002 | 0.023 | 0.192 | 1.000 |
| 3 | 4 | 0.002 | 0.022 | 0.190 | 1.000 |
| 4 | 4 | 0.001 | 0.018 | 0.181 | 1.000 |

**8 heads/layer** (removable before the effect passes 10% of full: 0.62 ± 0.00 of the induction heads)

| seed | n | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 8 | 0.000 | 0.002 | 0.005 | 0.023 | 0.061 | 0.154 | 0.434 | 1.000 |
| 1 | 8 | 0.000 | 0.001 | 0.004 | 0.009 | 0.037 | 0.145 | 0.392 | 1.000 |
| 2 | 8 | 0.000 | 0.001 | 0.005 | 0.011 | 0.062 | 0.136 | 0.396 | 1.000 |
| 3 | 8 | 0.000 | 0.001 | 0.004 | 0.012 | 0.056 | 0.140 | 0.392 | 1.000 |
| 4 | 8 | 0.000 | 0.002 | 0.004 | 0.016 | 0.055 | 0.147 | 0.412 | 1.000 |

## 4. Sufficiency: keep one head of a kind, ablate the others

Effect with one head left ÷ effect with none left (0 = one head is enough, 1 = one head does nothing).

| heads/layer | induction heads | prev-token heads |
|---|---|---|
| 2 | 0.06 ± 0.04 (n=12) | 0.07 ± 0.05 (n=2) |
| 4 | 0.23 ± 0.16 (n=16) | 0.06 ± 0.06 (n=2) |
| 8 | 0.41 ± 0.12 (n=40) | 0.10 ± 0.08 (n=7) |

## 5. Hidden backups outside the reference circuit

Non-circuit layer-0 heads (previous-token score below the 0.3 threshold). *Alone* = single-head effect; *after P* = effect once the head's strongest reference previous-token partner P is already ablated (effect(head and P) − effect(P)); both ÷ whole-circuit effect. Hidden = alone < 1%, after P > 1%, and the pair interaction's CI excludes 0. ρ = Spearman between previous-token score and the interaction with P (pooled over seeds; per-model median in brackets).

| heads/layer | non-circuit layer-0 heads | pass 1% alone | pass 1% after P | hidden | hidden, score ≥ 0.1 | hidden, score < 0.1 | median after P ÷ alone | ρ(prev-token score, interaction) |
|---|---|---|---|---|---|---|---|---|
| 2 | 5 | 0 | 4 | 4 | 1/1 | 3/4 | 666 | 0.90 (–) |
| 4 | 11 | 0 | 8 | 8 | 5/5 | 3/6 | 491 | 0.95 (1.00) |
| 8 | 31 | 0 | 19 | 19 | 18/19 | 1/12 | 485 | 0.86 (0.96) |
