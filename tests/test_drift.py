"""Cache-state drift: when the router's picture of the fleet stops being true."""

from kv_aware_router.policies import make_policy
from kv_aware_router.replay import replay
from kv_aware_router.workload import WorkloadSpec, generate

REPLICAS = ["a", "b", "c", "d"]
REQS = generate(WorkloadSpec(n_sessions=80, seed=3))
BACKGROUND = generate(WorkloadSpec(n_sessions=80, seed=99))
TIGHT = 6_000


def run(**kw):
    return replay(REQS, make_policy("cost_model"), REPLICAS,
                  concurrency=8, capacity_tokens=TIGHT, **kw)


def test_a_tighter_cache_serves_less_from_cache():
    roomy = replay(REQS, make_policy("cost_model"), REPLICAS,
                   concurrency=8, capacity_tokens=None)
    tight = run()
    assert tight.reuse_fraction < roomy.reuse_fraction
    assert tight.evictions > 0


def test_a_perfectly_modelled_cache_does_not_drift():
    # the router sees every request and knows the true capacity, so its
    # eviction model reproduces the replica exactly. Drift needs a cause.
    result = run()
    assert result.drift_fraction == 0.0
    assert result.mispredicted_requests == 0


def test_overestimating_capacity_causes_drift():
    assert run(belief_capacity_ratio=4.0).drift_fraction > 0.0


def test_unobserved_traffic_causes_drift():
    # another router evicting from the same replicas
    assert run(background=BACKGROUND).drift_fraction > 0.0


def test_drift_compounds_when_both_causes_are_present():
    only_capacity = run(belief_capacity_ratio=2.0).drift_fraction
    only_background = run(background=BACKGROUND).drift_fraction
    both = run(belief_capacity_ratio=2.0, background=BACKGROUND).drift_fraction
    assert both > only_capacity
    assert both > only_background


def test_assuming_infinite_memory_is_the_worst_case():
    blind = run(belief="blind").drift_fraction
    assert blind > run(belief_capacity_ratio=2.0).drift_fraction


def test_perfect_information_is_the_upper_bound_on_reuse():
    oracle = run(belief="oracle")
    modelled = run()
    assert oracle.reuse_fraction >= modelled.reuse_fraction
    assert oracle.drift_fraction == 0.0


def test_drift_only_counts_overprediction():
    # believing a replica is colder than it is costs reuse, not accuracy;
    # the metric is about the router being over-confident
    assert run(belief_capacity_ratio=0.5).drift_fraction == 0.0


def test_sticky_policies_mispredict_more_than_the_cost_model():
    # they route into evicted caches with no mechanism to notice
    kw = dict(concurrency=8, capacity_tokens=TIGHT,
              belief_capacity_ratio=2.0, background=BACKGROUND)
    sticky = replay(REQS, make_policy("consistent_hash"), REPLICAS, **kw)
    cost = replay(REQS, make_policy("cost_model"), REPLICAS, **kw)
    assert sticky.misprediction_rate > cost.misprediction_rate
