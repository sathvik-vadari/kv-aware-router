"""The service model, and the latency behaviour it exposes."""

from kv_aware_router.policies import make_policy
from kv_aware_router.simulate import ServiceModel, simulate
from kv_aware_router.workload import WorkloadSpec, generate

REPLICAS = ["a", "b", "c", "d"]
REQS = generate(WorkloadSpec(n_sessions=80, seed=3))


def run(name: str, **kw):
    return simulate(REQS, make_policy(name), REPLICAS, capacity_tokens=8000, **kw)


def test_every_request_completes():
    result = run("cost_model")
    assert len(result.outcomes) == len(REQS)
    assert sum(result.requests_per_replica.values()) == len(REQS)


def test_ttft_is_never_less_than_the_prefill_it_had_to_do():
    service = ServiceModel()
    for outcome in run("cost_model", service=service).outcomes:
        floor = outcome.uncached_tokens * service.prefill_s_per_token
        assert outcome.ttft_s >= floor - 1e-9


def test_ttft_equals_queue_wait_plus_service():
    for outcome in run("round_robin").outcomes:
        assert outcome.ttft_s >= outcome.queue_wait_s - 1e-9


def test_a_cache_hit_shortens_prefill():
    # later turns of a session reuse more, so they should not be slower
    result = run("pure_affinity", load_scale=0.25)   # near-zero queueing
    cold = [o.ttft_s for o in result.outcomes if o.cached_tokens == 0]
    warm = [o.ttft_s for o in result.outcomes if o.cached_tokens > 0]
    assert cold and warm
    assert sum(warm) / len(warm) < sum(cold) / len(cold)


def test_more_load_means_worse_tail():
    light = run("cost_model", load_scale=1.0)
    heavy = run("cost_model", load_scale=6.0)
    assert heavy.ttft(99) > light.ttft(99)
    assert heavy.queue_wait(99) > light.queue_wait(99)


def test_chasing_the_cache_blindly_costs_tail_latency():
    # pure_affinity keeps the warmest replica warm by overloading it
    affinity = run("pure_affinity", load_scale=4.0)
    balanced = run("round_robin", load_scale=4.0)
    assert affinity.reuse_fraction > balanced.reuse_fraction
    assert affinity.ttft(99) > balanced.ttft(99)


def test_cost_model_has_a_better_tail_than_the_affinity_policies():
    heavy = dict(load_scale=4.0)
    cost = run("cost_model", **heavy)
    for name in ("pure_affinity", "consistent_hash"):
        assert cost.ttft(99) < run(name, **heavy).ttft(99)


def test_cost_model_keeps_more_reuse_than_pure_balance():
    heavy = dict(load_scale=4.0)
    assert run("cost_model", **heavy).reuse_fraction > run("round_robin", **heavy).reuse_fraction
