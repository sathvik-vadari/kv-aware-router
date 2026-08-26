"""What replay measures, and what it refuses to claim."""

from kv_aware_router.policies import make_policy
from kv_aware_router.replay import replay
from kv_aware_router.workload import WorkloadSpec, generate, reuse_ceiling

REPLICAS = ["a", "b", "c", "d"]
REQS = generate(WorkloadSpec(n_sessions=60, seed=3))


def test_a_single_request_can_reuse_nothing():
    result = replay(REQS[:1], make_policy("pure_affinity"), REPLICAS)
    assert result.cached_tokens == 0
    assert result.reuse_fraction == 0.0


def test_no_policy_beats_the_oracle_ceiling():
    ceiling = reuse_ceiling(REQS)
    for name in ("round_robin", "pure_affinity", "cost_model", "consistent_hash"):
        result = replay(REQS, make_policy(name), REPLICAS, concurrency=8)
        assert result.reuse_fraction <= ceiling + 1e-9


def test_cache_aware_beats_cache_blind_on_reuse():
    blind = replay(REQS, make_policy("round_robin"), REPLICAS, concurrency=8)
    aware = replay(REQS, make_policy("pure_affinity"), REPLICAS, concurrency=8)
    assert aware.reuse_fraction > blind.reuse_fraction


def test_round_robin_is_perfectly_balanced():
    result = replay(REQS, make_policy("round_robin"), REPLICAS, concurrency=8)
    assert result.load_cv < 1e-9


def test_affinity_wins_reuse_but_loses_balance():
    aware = replay(REQS, make_policy("pure_affinity"), REPLICAS, concurrency=8)
    blind = replay(REQS, make_policy("round_robin"), REPLICAS, concurrency=8)
    assert aware.reuse_fraction > blind.reuse_fraction
    assert aware.load_cv > blind.load_cv


def test_cost_model_sits_between_the_two_extremes():
    rr = replay(REQS, make_policy("round_robin"), REPLICAS, concurrency=8)
    aff = replay(REQS, make_policy("pure_affinity"), REPLICAS, concurrency=8)
    cm = replay(REQS, make_policy("cost_model"), REPLICAS, concurrency=8)
    assert rr.reuse_fraction < cm.reuse_fraction < aff.reuse_fraction
    assert rr.load_cv < cm.load_cv < aff.load_cv


def test_every_request_is_accounted_to_exactly_one_replica():
    result = replay(REQS, make_policy("cost_model"), REPLICAS, concurrency=8)
    assert sum(result.requests_per_replica.values()) == len(REQS)
