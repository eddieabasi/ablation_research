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

* **Hypothesis**: `zero − mean` is predicted by the norm of the head's mean output
  relative to the residual stream, and is mediated by LayerNorm rescaling.
* **Experiment**: decompose zero effect = (zero → mean shift) + (mean ablation effect);
  regress on mean-output norm; repeat with LayerNorm frozen at clean statistics to test
  the mediation claim.

### Q3. Self-repair: when a circuit head is ablated, do other components compensate?

* **Hypothesis**: Total effect < direct effect for heads with backups; compensation is
  larger under mean than resample ablation.
* **Experiment**: compare total effect (ablate, let everything recompute) with direct
  effect (patch only the head→logits path). Needs ≥3 layers to have room for repair;
  also measure on the redundant 8-induction-head model.

### Q4. Redundancy: does single-component ablation miss components that only matter jointly?

* **Hypothesis**: In models with redundant heads, single ablations are near zero for every
  circuit head while joint ablation is large, so single-head attribution has low recall.
  Interaction `effect(A ∪ B) − effect(A) − effect(B)` is strongly positive for backup pairs.
* **Experiment**: pairwise joint ablations within the circuit; vary the number of heads to
  control redundancy.

### Q5. How well does attribution patching (a linear approximation) predict real ablation effects?

* **Hypothesis**: Good rank agreement for small, in-distribution interventions (resample
  of layer-1 heads), poor where LayerNorm or softmax saturation make the loss nonlinear
  (zero ablation, layer-0 heads that feed attention scores).
* **Experiment**: gradient × (ablated − clean activation) vs measured effect, per head,
  per method.

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
