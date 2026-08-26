"""How the five policies differ, on identical traffic."""

import pytest

from kv_aware_router.fleet import CostModel, Fleet
from kv_aware_router.policies import POLICIES, make_policy

TURN1 = tuple(range(750))                       # system prompt + turn 1 + reply
TURN2 = TURN1 + tuple(range(9000, 9050))        # same conversation, one turn later


def warm(replica: str = "a") -> Fleet:
    """A fleet where `replica` served turn 1 and finished it."""
    fleet = Fleet(["a", "b", "c"])
    d = fleet.dispatch(replica, TURN1, now=0.0)
    fleet.complete(d, now=1.0)
    return fleet


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_every_policy_returns_a_real_replica(name):
    fleet = warm()
    assert make_policy(name).choose(fleet, TURN2) in fleet.replicas


def test_round_robin_ignores_the_cache_entirely():
    fleet, policy = warm(), make_policy("round_robin")
    assert [policy.choose(fleet, TURN2) for _ in range(4)] == ["a", "b", "c", "a"]


def test_least_connections_prefers_idle_over_warm():
    fleet = Fleet(["a", "b"])
    fleet.dispatch("a", TURN1, now=0.0)          # a holds the cache and is busy
    assert make_policy("least_connections").choose(fleet, TURN2) == "b"


def test_pure_affinity_follows_the_cache():
    assert make_policy("pure_affinity").choose(warm(), TURN2) == "a"


def test_pure_affinity_follows_the_cache_even_into_a_meltdown():
    # the failure mode: it will keep piling work onto the warm replica no
    # matter how deep the queue gets
    fleet = warm()
    fleet.replicas["a"].in_flight = 100
    fleet.replicas["a"].pending_prefill_tokens = 500_000
    assert make_policy("pure_affinity").choose(fleet, TURN2) == "a"


def test_cost_model_uses_the_cache_when_the_fleet_is_idle():
    assert make_policy("cost_model").choose(warm(), TURN2) == "a"


def test_cost_model_abandons_the_cache_when_the_queue_costs_more():
    # the thesis: the same policy that chose affinity above chooses balance
    # here, with no threshold and no configuration change
    fleet = warm()
    fleet.replicas["a"].in_flight = 20
    fleet.replicas["a"].pending_prefill_tokens = 20_000
    assert make_policy("cost_model").choose(fleet, TURN2) != "a"


def test_cost_model_crossover_is_where_queue_delay_overtakes_saved_prefill():
    # a holds 736 of 800 tokens, so routing elsewhere costs 736 extra prefill
    # tokens; a should stay the pick until its queue exceeds roughly that
    policy = make_policy("cost_model", cost=CostModel(decode_interference_s=0.0))
    below, above = None, None
    for queued in range(0, 2000, 16):
        fleet = warm()
        fleet.replicas["a"].pending_prefill_tokens = queued
        pick = policy.choose(fleet, TURN2)
        if pick == "a":
            below = queued
        elif above is None:
            above = queued
    assert below is not None and above is not None
    assert 700 <= above <= 780        # crossover sits at the cached-token count


def test_consistent_hash_is_stable_across_turns_of_one_conversation():
    fleet = Fleet(["a", "b", "c"])
    policy = make_policy("consistent_hash")
    assert policy.choose(fleet, TURN1) == policy.choose(fleet, TURN2)


def test_consistent_hash_spreads_distinct_conversations():
    fleet = Fleet(["a", "b", "c"])
    policy = make_policy("consistent_hash")
    picks = {policy.choose(fleet, tuple(range(i, i + 200))) for i in range(0, 4000, 97)}
    assert len(picks) > 1          # not collapsing everything onto one replica


def test_consistent_hash_keys_on_the_session_not_the_prompt():
    fleet = Fleet(["a", "b", "c"])
    policy = make_policy("consistent_hash")
    # two unrelated sessions sharing a system prompt must be free to land on
    # different replicas; keying on the token prefix would pin them together
    shared_head = tuple(range(512))
    picks = {
        policy.choose(fleet, shared_head + (i,), session_key=f"s{i}")
        for i in range(30)
    }
    assert len(picks) > 1


def test_token_prefix_hashing_collapses_onto_the_system_prompt():
    # documents why the session key is required rather than optional: with no
    # key, sessions sharing a system prompt all hash to one replica, which is
    # "shard by system prompt" wearing consistent hashing's name
    fleet = Fleet(["a", "b", "c"])
    policy = make_policy("consistent_hash")
    shared_head = tuple(range(512))
    picks = {policy.choose(fleet, shared_head + (i,)) for i in range(30)}
    assert len(picks) == 1
