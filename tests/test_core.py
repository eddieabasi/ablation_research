import torch

from ablation import tasks as T
from ablation.ablate import (
    METHODS,
    AblationContext,
    ablation_effect,
    ablation_hooks,
    fit_optimal_constants,
    result_hook_name,
)
from ablation.circuit import head_scores
from ablation.model import Config, Transformer


def tiny_model(seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    m = Transformer(Config(vocab_size=64, n_ctx=56, d_model=32, n_layers=2, n_heads=4, d_head=8, n_tasks=T.N_TAGS))
    m.eval()
    return m


# --- tasks -------------------------------------------------------------------------


def test_induction_targets_follow_earlier_occurrence():
    task = T.InductionTask()
    b = task.sample(64, torch.Generator().manual_seed(0))
    assert (b.tokens[:, 0] == T.TAG_INDUCTION).all()
    for i in range(len(b)):
        toks = b.tokens[i].tolist()
        for t in torch.nonzero(b.mask[i]).flatten().tolist():
            earlier = [s for s in range(1, t) if toks[s] == toks[t]]
            assert earlier, "induction position must have an earlier occurrence"
            # every earlier occurrence is followed by the target (periodic sequence)
            assert all(toks[s + 1] == b.targets[i, t].item() for s in earlier)


def test_induction_first_period_is_unpredictable_and_distinct():
    task = T.InductionTask()
    b = task.sample(64, torch.Generator().manual_seed(1))
    for i in range(len(b)):
        first_masked = int(torch.nonzero(b.mask[i])[0])
        period = first_masked - 1
        block = b.tokens[i, 1 : period + 1].tolist()
        assert len(set(block)) == period
        assert task.min_period <= period <= task.max_period
        assert min(block) >= T.N_TAGS


def test_two_back_targets():
    task = T.TwoBackTask()
    b = task.sample(16, torch.Generator().manual_seed(0))
    t = torch.nonzero(b.mask[0]).flatten()
    assert t.min() == 3
    assert torch.equal(b.targets[:, t], b.tokens[:, t - 2])


def test_mixture_contains_both_tasks():
    b = T.Mixture().sample(256, torch.Generator().manual_seed(0))
    tags = set(b.tokens[:, 0].tolist())
    assert tags == {T.TAG_INDUCTION, T.TAG_TWO_BACK}


# --- model -------------------------------------------------------------------------


def test_head_results_sum_to_attention_output():
    """Per-head results must be exactly the heads' additive contributions."""
    m = tiny_model()
    b = T.InductionTask().sample(8, torch.Generator().manual_seed(0))
    _, cache = m.run_with_cache(b.tokens)
    attn = m.blocks[0].attn
    resid_after = cache["blocks.1.resid_pre"]
    resid_before = cache["blocks.0.resid_pre"]
    recon = resid_before + cache["blocks.0.attn.result"].sum(2) + attn.b_O
    assert torch.allclose(recon, resid_after, atol=1e-5)


def test_identity_hook_is_noop():
    m = tiny_model()
    b = T.InductionTask().sample(8, torch.Generator().manual_seed(0))
    with torch.no_grad():
        a = m(b.tokens)
        c = m(b.tokens, hooks={result_hook_name(0): lambda x: x.clone()})
    assert torch.equal(a, c)


def test_save_load_roundtrip(tmp_path):
    m = tiny_model()
    m.save(tmp_path / "m.pt")
    m2 = Transformer.load(tmp_path / "m.pt")
    b = T.InductionTask().sample(4, torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(m(b.tokens), m2(b.tokens))


# --- ablation ----------------------------------------------------------------------


def make_ctx(m, eval_b, ref_b, res_b):
    task = T.InductionTask()
    gen = torch.Generator().manual_seed(9)
    return AblationContext(m, eval_b, ref_b, res_b, lambda n: task.sample(n, gen))


def test_zero_ablation_zeroes_only_target_head():
    m = tiny_model()
    b = T.InductionTask().sample(8, torch.Generator().manual_seed(0))
    ctx = make_ctx(m, b, b, b)
    hooks = ablation_hooks(ctx, [(0, 2)], "zero")
    _, clean = m.run_with_cache(b.tokens)
    _, abl = m.run_with_cache(b.tokens, hooks=hooks)
    r_clean, r_abl = clean["blocks.0.attn.result"], abl["blocks.0.attn.result"]
    assert (r_abl[:, :, 2] == 0).all()
    keep = [0, 1, 3]
    assert torch.equal(r_abl[:, :, keep], r_clean[:, :, keep])


def test_mean_and_resample_are_noops_when_reference_is_the_input():
    """Mean over copies of one sequence = its own activation; resampling from itself = itself."""
    m = tiny_model()
    one = T.InductionTask().sample(1, torch.Generator().manual_seed(0))
    same = one[torch.zeros(6, dtype=torch.long)]
    ctx = make_ctx(m, same, same, same)
    for method in ("mean", "resample"):
        eff = ablation_effect(ctx, [(0, 1), (1, 3)], method)
        assert eff.abs().max() < 1e-5, method


def test_methods_differ_on_random_inputs():
    m = tiny_model()
    g = torch.Generator().manual_seed(0)
    task = T.InductionTask()
    ctx = make_ctx(m, task.sample(32, g), task.sample(64, g), task.sample(32, g))
    effects = {}
    for method in ("zero", "mean", "resample"):
        effects[method] = ablation_effect(ctx, [(0, 0)], method).mean().item()
    assert len({round(v, 6) for v in effects.values()}) == 3


def test_optimal_constants_do_not_exceed_mean_on_training_loss():
    m = tiny_model()
    g = torch.Generator().manual_seed(0)
    task = T.InductionTask()
    big = task.sample(512, g)
    ctx = make_ctx(m, big, task.sample(512, g), task.sample(512, g))
    mean_eff = ablation_effect(ctx, [(0, 0)], "mean").mean().item()
    consts = fit_optimal_constants(ctx, [(0, 0)], steps=60)
    opt_eff = ablation_effect(ctx, [(0, 0)], "optimal", constants=consts).mean().item()
    assert opt_eff <= mean_eff + 1e-3


def test_all_methods_run_on_groups_spanning_layers():
    m = tiny_model()
    g = torch.Generator().manual_seed(0)
    task = T.InductionTask()
    ctx = make_ctx(m, task.sample(16, g), task.sample(32, g), task.sample(16, g))
    heads = [(0, 1), (1, 0), (1, 2)]
    consts = fit_optimal_constants(ctx, heads, steps=5)
    for method in METHODS:
        eff = ablation_effect(ctx, heads, method, constants=consts if method == "optimal" else None)
        assert eff.shape == (16,) and torch.isfinite(eff).all()


# --- circuit scores ----------------------------------------------------------------


def test_head_scores_detect_a_hardwired_previous_token_pattern():
    m = tiny_model()
    b = T.InductionTask().sample(8, torch.Generator().manual_seed(0))
    n = b.tokens.shape[1]
    prev = torch.zeros(n, n)
    prev[0, 0] = 1
    prev[torch.arange(1, n), torch.arange(0, n - 1)] = 1

    def force_prev(p):
        p = p.clone()
        p[:, 0] = prev
        return p

    with torch.no_grad():
        cache: dict = {}
        m(b.tokens, hooks={"blocks.0.attn.pattern": force_prev}, cache=cache)
    # head_scores runs its own forward pass, so patch run_with_cache for the check
    orig = m.run_with_cache
    m.run_with_cache = lambda tokens, hooks=None, names=None: (None, cache)
    try:
        s = head_scores(m, b)
    finally:
        m.run_with_cache = orig
    assert s.prev_token[0, 0] > 0.999
    assert s.prev_token[0, 1:].max() < 0.9


def test_cache_names_limits_cache():
    m = tiny_model()
    b = T.InductionTask().sample(2, torch.Generator().manual_seed(0))
    _, cache = m.run_with_cache(b.tokens, names=["blocks.0.attn.result"])
    assert set(cache) == {"blocks.0.attn.result"}
