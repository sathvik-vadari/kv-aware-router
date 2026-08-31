"""Skewed session weight and replica churn: what a stateless hash cannot answer."""

from kv_aware_router.policies import make_policy
from kv_aware_router.simulate import ChurnEvent, simulate
from kv_aware_router.workload import WorkloadSpec, generate

REPLICAS = ["a", "b", "c", "d"]
SPEC = dict(n_sessions=120, turns_per_session=4, seed=5)
SKEWED = generate(WorkloadSpec(**SPEC, turn_skew=1.2))
SPAN = max(r.arrival_s for r in SKEWED)
LOAD = 3.0


def turns_per_session(requests) -> list[int]:
    counts: dict[int, int] = {}
    for r in requests:
        counts[r.session_id] = counts.get(r.session_id, 0) + 1
    return sorted(counts.values(), reverse=True)


def run(policy: str, churn=None):
    return simulate(SKEWED, make_policy(policy), REPLICAS,
                    capacity_tokens=8000, load_scale=LOAD, churn=churn)


def test_skew_zero_gives_every_session_the_same_length():
    counts = turns_per_session(generate(WorkloadSpec(**SPEC, turn_skew=0.0)))
    assert min(counts) == max(counts) == 4


def test_skew_produces_a_heavy_tail_without_moving_the_mean_much():
    counts = turns_per_session(SKEWED)
    assert max(counts) > 3 * 4                      # some very long sessions
    assert 3.0 < sum(counts) / len(counts) < 5.0    # mean roughly preserved


def test_a_removed_replica_stops_receiving_requests():
    result = run("round_robin", churn=[ChurnEvent(SPAN * 0.3, "remove", "d")])
    after = [o for o in result.outcomes if o.arrival_s >= (SPAN * 0.3) / LOAD]
    assert after
    assert all(o.replica != "d" for o in after)


def test_requests_in_flight_on_a_removed_replica_still_complete():
    result = run("round_robin", churn=[ChurnEvent(SPAN * 0.3, "remove", "d")])
    assert len(result.outcomes) == len(SKEWED)
    assert result.requests_per_replica["d"] > 0     # it served work before draining


def test_an_added_replica_receives_its_share():
    churn = [ChurnEvent(SPAN * 0.3, "add", "e")]
    result = run("round_robin", churn=churn)
    after = [o for o in result.outcomes if o.arrival_s >= (SPAN * 0.3) / LOAD]
    on_new = [o for o in after if o.replica == "e"]
    assert len(on_new) / len(after) > 0.1           # near the 1/5 fair share


def test_pure_affinity_starves_a_newly_added_replica():
    # it only ever picks the longest cached prefix, and a cold replica has none,
    # so scaling up adds capacity that receives nothing
    churn = [ChurnEvent(SPAN * 0.3, "add", "e")]
    result = run("pure_affinity", churn=churn)
    assert result.requests_per_replica.get("e", 0) == 0


def new_replica_share(policy: str, load: float) -> float:
    churn = [ChurnEvent(SPAN * 0.3, "add", "e")]
    result = simulate(SKEWED, make_policy(policy), REPLICAS,
                      capacity_tokens=8000, load_scale=load, churn=churn)
    after = [o for o in result.outcomes if o.arrival_s >= (SPAN * 0.3) / load]
    return len([o for o in after if o.replica == "e"]) / len(after)


def test_cost_model_also_starves_a_new_replica_at_moderate_load():
    # A real limitation, not a bug. Greedy per-request cost minimisation never
    # pays the one-off cost of warming a cold replica: each individual request
    # is genuinely faster on a warm one, so the new capacity is never explored.
    # This is why Preble's scheduler is built around exploitation *and
    # exploration* rather than pure exploitation.
    assert new_replica_share("cost_model", load=6.0) == 0.0


def test_cost_model_adopts_a_new_replica_once_queueing_dominates():
    # once the queue on the warm replicas costs more than a cold prefill, the
    # same policy switches over with nothing reconfigured
    assert new_replica_share("cost_model", load=24.0) > 0.1


def test_cost_model_stays_balanced_when_sessions_are_skewed():
    # a hash assigns by identity, so several heavy sessions can land together
    # with no way to rebalance; the cost model sees the queue and moves
    assert run("cost_model").load_cv < run("consistent_hash").load_cv


def test_pure_affinity_tail_collapses_under_skew():
    assert run("pure_affinity").ttft(99) > 1.5 * run("cost_model").ttft(99)
