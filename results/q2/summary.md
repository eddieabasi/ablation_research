# Q2 summary (5 seeds)

Effects are increases in induction loss (nats). `mean` removes a head's input-dependent information, `shift` removes its constant (mean) output, `zero` removes both. Interaction = zero − mean − shift. Shapley share of the constant = ½[shift + (zero − mean)] / zero.

## 1. Decomposing zero ablation

Groups (mean ± sd over seeds):

| group | zero | mean (info) | shift (constant) | interaction | constant's Shapley share |
|---|---|---|---|---|---|
| non-circuit heads | 0.71 ± 0.35 | 0.026 ± 0.029 | 0.212 ± 0.070 | 0.47 ± 0.30 | 0.66 ± 0.08 |
| prev-token heads | 3.58 ± 2.30 | 2.741 ± 2.307 | 0.031 ± 0.027 | 0.81 ± 0.32 | 0.17 ± 0.09 |
| induction heads | 5.34 ± 0.06 | 5.339 ± 0.050 | 0.000 ± 0.000 | 0.00 ± 0.04 | 0.00 ± 0.00 |
| all of layer 0 | 6.19 ± 0.21 | 6.662 ± 0.135 | 0.575 ± 0.276 | -1.04 ± 0.26 | 0.01 ± 0.03 |

Single heads (median [IQR], pooled over seeds):

| head type | n | zero | mean | shift | interaction | constant's Shapley share |
|---|---|---|---|---|---|---|
| prev-token | 9 | 0.3539 [0.2299, 1.0266] | 0.1451 [0.0578, 0.4466] | 0.0050 [0.0047, 0.0054] | 0.2455 [0.1677, 0.4915] | 0.34 [0.28, 0.38] |
| induction | 40 | 0.0014 [0.0009, 0.0021] | 0.0013 [0.0009, 0.0020] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0001] | 0.02 [0.01, 0.03] |
| non-circuit | 31 | 0.0054 [0.0025, 0.0068] | 0.0003 [0.0001, 0.0004] | 0.0032 [0.0022, 0.0050] | 0.0008 [0.0002, 0.0019] | 0.83 [0.77, 0.90] |

## 2. Does LayerNorm mediate it? (effect with LayerNorm frozen at clean scale ÷ live effect)

Ratio > 1: freezing LayerNorm makes the ablation *worse*, i.e. live LayerNorm was cushioning it. Only ablations with a live effect above 0.001 nats are included (ratios of near-zero effects are noise).

| unit | zero | mean | shift |
|---|---|---|---|
| non-circuit heads (group) | 1.28 ± 0.16 | 5.42 ± 1.79 | 1.08 ± 0.18 |
| prev-token heads (group) | 1.11 ± 0.34 | 1.59 ± 0.65 | 1.70 ± 0.51 |
| induction heads (group) | 0.79 ± 0.01 | 0.79 ± 0.01 | – |
| all of layer 0 (group) | 0.95 ± 0.06 | 0.74 ± 0.02 | 1.09 ± 0.16 |
| prev-token (single, median [IQR]) | 2.04 [1.62, 2.53] (n=9) | 4.07 [3.00, 5.58] (n=9) | 1.93 [1.55, 2.15] (n=9) |
| induction (single, median [IQR]) | 3.36 [2.98, 3.74] (n=26) | 3.40 [3.01, 3.78] (n=26) | – |
| non-circuit (single, median [IQR]) | 1.63 [1.13, 2.05] (n=31) | 6.69 [6.69, 6.69] (n=1) | 1.21 [1.00, 1.50] (n=30) |

Fraction of single-head ablations (effect > 0.001) made worse by freezing LayerNorm: zero 95%, mean 100%, shift 79%.

## 3. Which downstream reads carry the damage? (layer-0 groups; path effect ÷ full effect)

Each column delivers the ablation to one receiver only: layer-1 queries (q), keys (k), values (v), the unembedding directly (direct), or queries and keys together (q+k).

| unit | method | full effect | q | k | v | direct | q+k |
|---|---|---|---|---|---|---|---|
| non-circuit heads | zero | 0.71 ± 0.35 | 0.31 ± 0.27 | 0.14 ± 0.11 | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.89 ± 0.08 |
| non-circuit heads | mean | 0.03 ± 0.03 | 0.27 ± 0.33 | 0.52 ± 0.40 | 0.00 ± 0.00 | -0.00 ± 0.00 | 1.30 ± 0.11 |
| non-circuit heads | shift | 0.21 ± 0.07 | 0.39 ± 0.20 | 0.08 ± 0.03 | 0.04 ± 0.02 | 0.00 ± 0.00 | 0.75 ± 0.15 |
| prev-token heads | zero | 3.58 ± 2.30 | 0.00 ± 0.00 | 1.01 ± 0.05 | 0.00 ± 0.00 | -0.00 ± 0.00 | 1.05 ± 0.04 |
| prev-token heads | mean | 2.74 ± 2.31 | -0.00 ± 0.00 | 1.12 ± 0.09 | 0.00 ± 0.00 | -0.00 ± 0.00 | 1.10 ± 0.06 |
| prev-token heads | shift | 0.03 ± 0.03 | 0.10 ± 0.02 | 0.46 ± 0.34 | 0.04 ± 0.01 | 0.00 ± 0.00 | 0.87 ± 0.05 |
| all of layer 0 | zero | 6.19 ± 0.21 | 0.06 ± 0.05 | 1.05 ± 0.03 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.10 ± 0.02 |
| all of layer 0 | mean | 6.66 ± 0.14 | 0.00 ± 0.00 | 0.95 ± 0.01 | 0.00 ± 0.00 | -0.00 ± 0.00 | 0.97 ± 0.01 |
| all of layer 0 | shift | 0.58 ± 0.28 | 0.31 ± 0.13 | 0.08 ± 0.04 | 0.03 ± 0.02 | 0.00 ± 0.00 | 0.82 ± 0.18 |

## 4. What the induction heads do under each ablation (groups, mean ± sd over seeds)

Clean: induction attention 0.88 ± 0.01, output entropy 0.009 ± 0.000 nats, accuracy 0.9999 ± 0.0000.

Induction attention = layer-1 attention mass on the key holding the answer, averaged over heads and induction positions. LN ratios = LayerNorm scale relative to clean (1 by construction when frozen).

| unit | condition | induction attention | output entropy | accuracy | layer-1 LN scale | final LN scale |
|---|---|---|---|---|---|---|
| non-circuit heads | zero | 0.57 ± 0.07 | 0.59 ± 0.26 | 0.85 ± 0.07 | 1.06 ± 0.04 | 0.93 ± 0.03 |
| non-circuit heads | zero, LN frozen | 0.58 ± 0.09 | 0.75 ± 0.29 | 0.81 ± 0.09 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| non-circuit heads | mean | 0.79 ± 0.05 | 0.03 ± 0.02 | 0.99 ± 0.01 | 0.94 ± 0.02 | 1.00 ± 0.03 |
| non-circuit heads | mean, LN frozen | 0.71 ± 0.09 | 0.25 ± 0.24 | 0.97 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| non-circuit heads | shift | 0.71 ± 0.02 | 0.28 ± 0.06 | 0.95 ± 0.02 | 1.10 ± 0.03 | 0.94 ± 0.02 |
| non-circuit heads | shift, LN frozen | 0.75 ± 0.03 | 0.24 ± 0.05 | 0.95 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| prev-token heads | zero | 0.26 ± 0.16 | 0.83 ± 0.33 | 0.44 ± 0.30 | 0.95 ± 0.01 | 0.83 ± 0.06 |
| prev-token heads | zero, LN frozen | 0.20 ± 0.12 | 1.97 ± 0.33 | 0.34 ± 0.25 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| prev-token heads | mean | 0.30 ± 0.19 | 0.68 ± 0.38 | 0.55 ± 0.34 | 0.90 ± 0.01 | 0.82 ± 0.10 |
| prev-token heads | mean, LN frozen | 0.20 ± 0.14 | 2.04 ± 0.41 | 0.39 ± 0.29 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| prev-token heads | shift | 0.81 ± 0.04 | 0.05 ± 0.03 | 0.99 ± 0.01 | 1.04 ± 0.02 | 0.95 ± 0.03 |
| prev-token heads | shift, LN frozen | 0.83 ± 0.03 | 0.07 ± 0.04 | 0.99 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| induction heads | zero | 0.88 ± 0.01 | 2.81 ± 0.03 | 0.02 ± 0.00 | 1.00 ± 0.00 | 0.30 ± 0.01 |
| induction heads | zero, LN frozen | 0.88 ± 0.01 | 4.00 ± 0.01 | 0.02 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| induction heads | mean | 0.88 ± 0.01 | 2.80 ± 0.03 | 0.02 ± 0.00 | 1.00 ± 0.00 | 0.29 ± 0.01 |
| induction heads | mean, LN frozen | 0.88 ± 0.01 | 4.01 ± 0.01 | 0.02 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| induction heads | shift | 0.88 ± 0.01 | 0.01 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| induction heads | shift, LN frozen | 0.88 ± 0.01 | 0.01 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| all of layer 0 | zero | 0.06 ± 0.01 | 1.75 ± 0.24 | 0.06 ± 0.01 | 1.04 ± 0.01 | 0.89 ± 0.06 |
| all of layer 0 | zero, LN frozen | 0.06 ± 0.01 | 2.13 ± 0.13 | 0.04 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| all of layer 0 | mean | 0.06 ± 0.00 | 1.07 ± 0.04 | 0.08 ± 0.00 | 0.86 ± 0.01 | 0.69 ± 0.02 |
| all of layer 0 | mean, LN frozen | 0.05 ± 0.00 | 2.62 ± 0.16 | 0.07 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| all of layer 0 | shift | 0.64 ± 0.05 | 0.49 ± 0.07 | 0.88 ± 0.06 | 1.15 ± 0.01 | 0.91 ± 0.04 |
| all of layer 0 | shift, LN frozen | 0.69 ± 0.06 | 0.40 ± 0.07 | 0.88 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 |

## 5. What predicts the damage from removing a layer-0 head's constant? (Spearman ρ, pooled)

`q_read` / `k_read` / `v_read`: how far the head's mean output moves the layer-1 queries / keys / values (first-order, through the clean LayerNorm scale). `mean_norm`: size of the mean output itself.

| heads | n | target | mean_norm | q_read | k_read | v_read |
|---|---|---|---|---|---|---|
| non-circuit | 31 | shift | 0.60 | 0.74 | 0.60 | 0.49 |
| non-circuit | 31 | zero_minus_mean | 0.59 | 0.73 | 0.62 | 0.44 |
| all-layer0 | 40 | shift | 0.65 | 0.64 | 0.64 | 0.53 |
| all-layer0 | 40 | zero_minus_mean | 0.76 | 0.50 | 0.81 | 0.51 |
