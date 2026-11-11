# Scope: when can we trust an ablation?

## Motivation

Interpretability claims of the form "component X matters for behaviour Y" almost always
rest on an ablation: replace X's output with something else and measure how much Y
degrades. There are several ways to "replace with something else" (zero, mean, resample,
optimal constant), and in real models they often give different answers. In a pretrained
LLM we cannot tell which answer is right, because we don't know the true mechanism.

This project answers ablation-methodology questions in **toy transformers trained from
scratch on tasks whose mechanism is known**, so every ablation conclusion can be checked
against a reference that does not itself come from ablation.

## Testbed

* **Model**: 2-layer attention-only transformer, 8 heads/layer, d_model 64, pre-LayerNorm,
  learned positional embeddings, task embedding at every position. Trains in ~1.5 min on CPU.
* **Tasks** (50/50 mixture, tagged at position 0):
  * *Induction*: periodic random token sequences with a random period. Solved by layer-0
    previous-token heads feeding layer-1 induction heads.
  * *Two-back*: predict the token two positions back. Gives the model heads that are
    active on induction inputs but irrelevant to induction, the realistic situation where
    a component serves some other behaviour.
* **Reference circuit** (no ablation involved): heads identified from attention patterns
  (previous-token score in layer 0, induction score in later layers).
* **Ablation methods** (`src/ablation/ablate.py`): zero, mean, resample, optimal constant.
* **Metric**: increase in induction loss (nats) relative to the clean model, with
  bootstrap 95% CIs over sequences; 5 seeds per configuration.

Design decisions recorded along the way:

| Decision | Why |
|---|---|
| Random period in the induction task | With a fixed period a positional shortcut ("copy from t−D+1") solves the task without induction. Observed: no loss plateau, loss ≈ 0 by step 250. |
| Two-task mixture | Trained on induction alone, every head joined the circuit (all 8 layer-1 heads became induction heads), leaving no ground-truth-irrelevant heads to test false positives. |
| Task embedding at every position | With only a tag token at position 0, several layer-0 heads learned to attend to the tag to broadcast task identity, which made them partly relevant to induction and blurred the reference. |

## Questions

Each question has a falsifiable hypothesis and a planned experiment. Status is tracked
in the README.

### Q1. Do zero, mean, resample and optimal ablation agree on which heads matter, and which one recovers the known circuit?

* **Hypothesis**: All methods agree on the strongest components, but differ by an order of
  magnitude in effect size. Zero ablation produces false positives on heads with large
  constant outputs; resample ablation inflates effects by injecting confidently wrong
  information.
* **Experiment**: single-head and group ablations under all four methods, 5 seeds.
  Rank agreement (Spearman), circuit recovery (AUROC vs reference circuit), and joint
  ablation of all non-circuit heads (false-positive test).

### Q2. How much of a zero-ablation effect is the mean shift rather than information?

* **Hypothesis** (as first written): `zero − mean` is predicted by the norm of the head's
  mean output relative to the residual stream, and is mediated by LayerNorm rescaling.
* **Experiment** (as run, `experiments/q2_mean_shift.py`): a 2×2 design on each head's
  output r = μ + ε: `mean` (r → μ, removes information), `shift` (r → ε, removes the
  constant), `zero` (r → 0, removes both); interaction = zero − mean − shift. Each is
  repeated with every LayerNorm frozen at its clean scale, and, for layer-0 heads,
  path-specifically (the change delivered only to layer-1 queries, keys, values, or the
  unembedding). Predictors: mean-output norm vs how far the mean output moves the
  layer-1 queries/keys/values. Diagnostics: induction heads' attention on the answer key,
  output entropy, LayerNorm scale.
* **Leads from Q1**: across 80 heads, log(zero/mean effect) vs the head's constant
  fraction gives ρ = 0.90. Also unexplained: mean-ablating all of layer 0 is *worse*
  than zero-ablating it (5/5 seeds).

### Q3. Self-repair: when a circuit head is ablated, do other components compensate?

* **Hypothesis**: Total effect < direct effect for heads with backups; compensation is
  larger under mean than resample ablation.
* **Leads from Q2**: freezing LayerNorm makes 95–100% of single-head ablations worse
  (1.2–4×), so LayerNorm rescaling is one compensation mechanism to quantify.
* **Leads from Q4**: weak previous-token heads (attention score 0.1–0.3) do nothing while
  the main previous-token head is intact, but matter once it is ablated (median 3.6%,
  up to 37% of the circuit effect). That is a concrete case for Q3: is this backup a
  fixed spare (the induction heads always read it, and it only becomes decisive), or
  real compensation (the induction heads shift to it when the main signal disappears)?
  Path patching the induction heads' keys with and without the main head would tell.
* **Experiment**: compare total effect (ablate, let everything recompute) with direct
  effect (patch only the head→logits path). Needs ≥3 layers to have room for repair;
  also measure on the redundant 8-induction-head model.

### Q4. Redundancy: does single-component ablation miss components that only matter jointly?

* **Hypothesis**: In models with redundant heads, single ablations are near zero for every
  circuit head while joint ablation is large, so single-head attribution has low recall.
  Interaction `effect(A ∪ B) − effect(A) − effect(B)` is strongly positive for backup pairs.
* **Experiment** (as run, `experiments/q4_redundancy.py`): models with 2, 4 and 8 heads
  per layer (8 = the Q1 models; 2 and 4 trained with the same recipe *to criterion*: at
  least 1,500 steps, then until both tasks reach 99%, up to 10,000). For each: every single
  head, every pair of heads (interaction with bootstrap CI, plus the margin test: does the
  interaction exist in the target-logit margin or only in the loss?), nested ablation of k
  of the n induction heads, and sufficiency (keep one head of a kind). Mean ablation
  (primary) and resample ablation. Prediction: backup pairs (induction + induction)
  superadditive; serial pairs (previous-token + induction) subadditive.
* **Excluded**: 1 head per layer (learns two-back but not induction: 53% accuracy after
  5,000 steps, no previous-token head), and any model that misses the criterion. Narrow
  models often never leave the induction plateau: every model that met the criterion did
  so by step 1,500, and none that missed it recovered by 10,000. So extra 2-head seeds
  (5–34) were capped at 1,500 steps.
* **Leads from Q1**: no single induction head passes a 1%-of-circuit threshold under any
  method, in any seed, while the 8 together account for the whole effect.
* **Leads from Q2**: ablation effects are superadditive near the clean model and
  subadditive far from it; the Q2 margin test shows most of that superadditivity is inside
  the network, not loss curvature.
* **Result**: see [`results/q4/REPORT.md`](../results/q4/REPORT.md). Also found: *hidden
  backups*, layer-0 heads under the 0.3 previous-token threshold that matter only once the
  main previous-token head is ablated.

### Q5. How well does attribution patching (a linear approximation) predict real ablation effects?

* **Hypothesis**: Good rank agreement for small, in-distribution interventions (resample
  of layer-1 heads), poor where LayerNorm or softmax saturation make the loss nonlinear
  (zero ablation, layer-0 heads that feed attention scores).
* **Experiment**: gradient × (ablated − clean activation) vs measured effect, per head,
  per method.
* **Leads from Q4**: a linear approximation is additive by construction, so it cannot see
  the redundancy Q4 measured (8 induction heads that matter only jointly). Compare
  attribution patching on single heads and on whole groups.

### Q6. How sensitive is resample ablation to the choice of corrupted distribution?

* **Hypothesis**: Effect sizes vary several-fold between corruptions that break everything
  (unrelated random sequence) and minimal-pair corruptions that break only the repeat,
  and the ranking of heads can change.
* **Experiment**: resample from (a) another induction sequence, (b) the same sequence with
  the repeat destroyed, (c) a two-back sequence; compare effects and rankings.

## Out of scope (for now)

* Pretrained models (huggingface.co is blocked in this environment and there is no GPU).
  The toolkit is model-agnostic in principle; adding a loader would mean allowing that host.
* Neuron- and direction-level ablations (heads only, for now).
* MLP layers (the testbed is attention-only so the reference circuit is unambiguous).
