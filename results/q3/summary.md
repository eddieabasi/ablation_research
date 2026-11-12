# Q3 summary

Changes in target logit difference (logits) from mean ablation unless stated; negative = damage, positive = repair. *Direct* = the ablated head's own change in contribution; *responses* = change in every later head's contribution; *LN* = the final LayerNorm's rescaling. total = direct + responses + LN exactly. Ratios use heads whose direct (or gross) damage is at least 0.1 logits.

## Models

| model | seeds | prev-token heads | induction heads | clean logit diff | self-check (max change of unaffected heads) |
|---|---|---|---|---|---|
| 2 layers, 2 heads | 3 (0, 14, 16) | 1.3 ± 0.6 | 2.0 ± 0.0 | 13.4 ± 0.2 | 0.0e+00 |

## 1. One induction head ablated: how much of its direct damage is repaired, and by what

Shares of the head's direct damage (−direct). LN = final LayerNorm; responses = heads in later layers (3-layer models: layer 2). Repaired = (total − direct) ÷ −direct.

| model | method | heads | median direct | LN share | responses share | repaired | heads with repaired > 0 |
|---|---|---|---|---|---|---|---|
| 2 layers, 2 heads | mean | 6 | -6.69 | 0.39 | 0.00 | 0.39 | 100% |
| 2 layers, 2 heads | resample | 6 | -6.69 | 0.08 | 0.00 | 0.08 | 100% |

## 2. One previous-token head ablated: the damage is indirect

Responses split by layer. LN1 cushion = share of the responses' damage that the attention-input LayerNorms prevent: 1 − responses ÷ responses with those LayerNorms frozen.

| model | heads | median direct | median responses: layer 1 | median responses: layer 2 | median LN | median total | median LN1 cushion |
|---|---|---|---|---|---|---|---|
| 2 layers, 2 heads | 4 | -0.04 | -6.38 | – | -0.05 | -6.47 | 0.41 |

## 3. Every head: gross damage vs repair, and where the repair comes from

Gross damage D = sum of negative components (direct, each response, LN); repair R = sum of positive ones. Source shares are shares of total R summed over heads.

| model | method | heads with D ≥ 0.1 | median R ÷ D | R from final LN | R from later heads | R from the head itself |
|---|---|---|---|---|---|---|
| 2 layers, 2 heads | mean | 12 | 0.38 | 97% | 3% | 0% |
| 2 layers, 2 heads | resample | 12 | 0.08 | 100% | 0% | 0% |

## 4. Mean vs resample ablation

Per head (D ≥ 0.1 under both): R ÷ D under each method.

| model | heads | median R ÷ D, mean | median R ÷ D, resample | share of heads where mean > resample |
|---|---|---|---|---|
| 2 layers, 2 heads | 12 | 0.38 | 0.08 | 92% |

## 5. Whole groups: repair that works for one head fails for all of them

Logit-diff interaction = total(group) − Σ total(single heads). Because direct effects add exactly, the interaction is exactly (LN(group) − Σ LN(singles)) + (responses(group) − Σ responses(singles)).

| model | group | heads | total (group) | Σ totals (singles) | interaction | from LN | from responses | LN share of direct: singles → group |
|---|---|---|---|---|---|---|---|---|
| 2 layers, 2 heads | induction_heads | 2.0 ± 0.0 | -13.0 ± 0.2 | -8.0 ± 0.3 | -5.0 ± 0.6 | -5.0 ± 0.6 | 0.0 ± 0.0 | 0.40 ± 0.03 → 0.02 ± 0.00 |
| 2 layers, 2 heads | prev_token_heads | 1.3 ± 0.6 | -11.2 ± 0.5 | -8.5 ± 4.4 | -2.7 ± 4.7 | 0.6 ± 1.0 | -3.3 ± 5.7 | – → – |
| 2 layers, 2 heads | layer1 | 2.0 ± 0.0 | -13.0 ± 0.2 | -8.0 ± 0.3 | -5.0 ± 0.6 | -5.0 ± 0.6 | 0.0 ± 0.0 | 0.40 ± 0.03 → 0.02 ± 0.00 |

## 6. Backup pairs with LayerNorms frozen: static redundancy or compensation?

Margin interaction (nats; > 0 superadditive) of same-layer pairs, median over pairs, with every LayerNorm free, the final one frozen, the attention-input ones frozen, or all frozen. Kept = Σ interaction frozen ÷ Σ interaction free (pooled over pairs).

| model | method | pair type | n | free | final frozen | attn-input frozen | all frozen | kept: final | kept: attn-input | kept: all |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 layers, 2 heads | mean | ind-ind | 3 | 4.054 | -0.116 | 4.054 | -0.116 | -4% | 100% | -4% |
| 2 layers, 2 heads | mean | prev-weak | 1 | 1.717 | 1.801 | 0.358 | -0.031 | 105% | 21% | -2% |
| 2 layers, 2 heads | resample | ind-ind | 3 | 3.927 | 3.832 | 3.927 | 3.832 | 97% | 100% | 97% |
| 2 layers, 2 heads | resample | prev-weak | 1 | 0.132 | -0.082 | 0.106 | -0.293 | -62% | 81% | -222% |

## 7. Loss: total effect vs direct path only

Heads not in the last layer. Direct path = the head's change delivered only to the unembedding (path-specific ablation). Total < direct path means the rest of the network repairs more than it adds.

| model | role | heads | median loss: total | median loss: direct path | share with total < direct path |
|---|---|---|---|---|---|
| 2 layers, 2 heads | prev | 4 | 3.429 | -0.000 | 0% |
| 2 layers, 2 heads | non | 2 | 0.001 | -0.000 | 50% |
