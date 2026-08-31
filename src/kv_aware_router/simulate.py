"""Discrete-event simulation of the fleet, so latency becomes measurable.

The replay harness measures work and balance. It cannot measure latency, and
that matters because the cost model optimises `queue_delay + prefill_cost` --
judging it on prefill alone scores half its objective.

Service model, and its assumptions stated plainly:

  prefill is the serialised resource
      One prefill at a time per replica. Prefill saturates the GPU's arithmetic
      units, so a second concurrent prefill does not go faster, it interleaves.
      This is what makes queueing -- and therefore TTFT -- happen at all.

  decode overlaps
      Once prefilled, a request joins the decode batch and no longer blocks the
      prefill queue. That is continuous batching, roughly.

  decode slows prefill
      Every concurrent decode steals memory bandwidth and a share of each step,
      so prefill duration scales with how many requests are decoding.

  TTFT = queue wait + prefill of the uncached tokens

What it ignores: chunked prefill interleaving at token granularity, decode step
time growing with batch size, KV cache pressure causing preemption, and the
quadratic term in attention. Each is a reason a measured TTFT would differ.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

from .fleet import Dispatch, Fleet
from .policies import Policy
from .radix import PrefixTree
from .replay import BeliefMode
from .workload import Request


@dataclass(frozen=True, slots=True)
class ServiceModel:
    """Timing constants. Defaults are order-of-magnitude for an 8B model on one
    datacentre GPU; calibrating them against a real backend is its own task."""

    prefill_s_per_token: float = 1e-4      # ~10k tokens/s
    decode_s_per_token: float = 8e-3       # ~125 tokens/s per request
    decode_interference: float = 0.05      # each concurrent decode slows prefill 5%
    output_tokens: int = 160


@dataclass(slots=True)
class Outcome:
    replica: str
    arrival_s: float
    prefill_start_s: float
    ttft_s: float
    finish_s: float
    cached_tokens: int
    uncached_tokens: int
    believed_cached_tokens: int

    @property
    def queue_wait_s(self) -> float:
        return self.prefill_start_s - self.arrival_s


@dataclass(slots=True)
class SimResult:
    policy: str
    load_scale: float
    capacity_tokens: int | None
    outcomes: list[Outcome] = field(default_factory=list)
    evictions: int = 0
    requests_per_replica: dict[str, int] = field(default_factory=dict)

    def _pct(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        k = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
        return ordered[k]

    def ttft(self, p: float) -> float:
        return self._pct([o.ttft_s for o in self.outcomes], p)

    def queue_wait(self, p: float) -> float:
        return self._pct([o.queue_wait_s for o in self.outcomes], p)

    @property
    def reuse_fraction(self) -> float:
        total = sum(o.cached_tokens + o.uncached_tokens for o in self.outcomes)
        return sum(o.cached_tokens for o in self.outcomes) / total if total else 0.0

    @property
    def drift_fraction(self) -> float:
        total = sum(o.cached_tokens + o.uncached_tokens for o in self.outcomes)
        over = sum(
            max(0, o.believed_cached_tokens - o.cached_tokens) for o in self.outcomes
        )
        return over / total if total else 0.0

    @property
    def load_cv(self) -> float:
        counts = list(self.requests_per_replica.values())
        if not counts:
            return 0.0
        mean = sum(counts) / len(counts)
        if mean == 0:
            return 0.0
        var = sum((c - mean) ** 2 for c in counts) / len(counts)
        return var**0.5 / mean


@dataclass(frozen=True, slots=True)
class ChurnEvent:
    """A replica joining or leaving mid-run, i.e. autoscaling.

    Removal drains: the replica stops receiving new requests and loses its
    cache, but work already queued on it finishes. That is what a graceful
    scale-down does.

    Churn is the condition a stateless session hash cannot answer. Changing the
    replica set reshuffles part of its keyspace, so sessions are reassigned to
    replicas that have never seen them -- and it has no way to know that, since
    it never looks at cache state.
    """

    at_s: float
    action: str          # "add" | "remove"
    replica_id: str


# event kinds, ordered so ties resolve deterministically
_CHURN, _ARRIVAL, _PREFILL_DONE, _DECODE_DONE = 0, 1, 2, 3


def simulate(
    requests: list[Request],
    policy: Policy,
    replica_ids: list[str],
    *,
    service: ServiceModel = ServiceModel(),
    match_unit: int = 16,
    capacity_tokens: int | None = None,
    belief: BeliefMode = "modelled",
    belief_capacity_ratio: float = 1.0,
    load_scale: float = 1.0,
    churn: list[ChurnEvent] | None = None,
) -> SimResult:
    """Run the workload through the fleet in simulated time.

    load_scale compresses arrivals: 2.0 means twice the request rate on the same
    traffic, which is how the load axis is swept.
    """
    truth = PrefixTree(
        prefix_match_unit=match_unit,
        capacity_tokens=(
            {rid: capacity_tokens for rid in replica_ids} if capacity_tokens else None
        ),
    )
    belief_capacity = (
        {rid: max(1, int(capacity_tokens * belief_capacity_ratio)) for rid in replica_ids}
        if capacity_tokens and belief == "modelled"
        else None
    )
    fleet = Fleet(replica_ids, prefix_match_unit=match_unit, capacity_tokens=belief_capacity)
    if belief == "oracle":
        fleet.tree = truth

    result = SimResult(
        policy=policy.name,
        load_scale=load_scale,
        capacity_tokens=capacity_tokens,
        requests_per_replica={rid: 0 for rid in replica_ids},
    )

    busy: dict[str, bool] = {rid: False for rid in replica_ids}
    pending: dict[str, deque] = {rid: deque() for rid in replica_ids}
    decoding: dict[str, int] = {rid: 0 for rid in replica_ids}

    events: list[tuple[float, int, int, object]] = []
    seq = 0
    for req in requests:
        heapq.heappush(events, (req.arrival_s / load_scale, seq, _ARRIVAL, req))
        seq += 1
    for event in churn or []:
        heapq.heappush(events, (event.at_s / load_scale, seq, _CHURN, event))
        seq += 1

    def start_next(replica: str, now: float) -> None:
        nonlocal seq
        if busy[replica] or not pending[replica]:
            return
        outcome, dispatch = pending[replica].popleft()
        busy[replica] = True
        outcome.prefill_start_s = now
        duration = (
            outcome.uncached_tokens
            * service.prefill_s_per_token
            * (1.0 + service.decode_interference * decoding[replica])
        )
        outcome.ttft_s = (now - outcome.arrival_s) + duration
        heapq.heappush(
            events, (now + duration, seq, _PREFILL_DONE, (replica, outcome, dispatch))
        )
        seq += 1

    while events:
        now, _, kind, payload = heapq.heappop(events)

        if kind == _CHURN:
            event: ChurnEvent = payload  # type: ignore[assignment]
            rid = event.replica_id
            if event.action == "add":
                fleet.add_replica(rid)
                busy.setdefault(rid, False)
                pending.setdefault(rid, deque())
                decoding.setdefault(rid, 0)
                result.requests_per_replica.setdefault(rid, 0)
                if capacity_tokens:
                    truth.capacity_tokens[rid] = capacity_tokens
                    if belief_capacity is not None:
                        fleet.tree.capacity_tokens[rid] = belief_capacity[rid] if rid in belief_capacity else int(capacity_tokens * belief_capacity_ratio)
            else:
                # stop routing here and drop the cache, but let queued work finish
                fleet.remove_replica(rid)
                truth.remove_replica(rid)
            continue

        if kind == _ARRIVAL:
            req: Request = payload  # type: ignore[assignment]
            replica = policy.choose(fleet, req.tokens, session_key=str(req.session_id))
            cached = truth.match(req.tokens).get(replica, 0)
            believed = fleet.tree.match(req.tokens).get(replica, 0)
            outcome = Outcome(
                replica=replica,
                arrival_s=now,
                prefill_start_s=now,
                ttft_s=0.0,
                finish_s=0.0,
                cached_tokens=cached,
                uncached_tokens=req.n_tokens - cached,
                believed_cached_tokens=believed,
            )
            result.requests_per_replica[replica] += 1
            truth.insert(req.tokens, replica, now=now)
            dispatch = fleet.dispatch(replica, req.tokens, now=now)
            pending[replica].append((outcome, dispatch))
            start_next(replica, now)

        elif kind == _PREFILL_DONE:
            replica, outcome, dispatch = payload  # type: ignore[misc]
            busy[replica] = False
            decoding[replica] += 1
            decode_s = service.output_tokens * service.decode_s_per_token
            heapq.heappush(
                events, (now + decode_s, seq, _DECODE_DONE, (replica, outcome, dispatch))
            )
            seq += 1
            start_next(replica, now)

        else:  # _DECODE_DONE
            replica, outcome, dispatch = payload  # type: ignore[misc]
            decoding[replica] -= 1
            outcome.finish_s = now
            fleet.complete(dispatch, now=now)
            result.outcomes.append(outcome)
            start_next(replica, now)

    result.evictions = sum(truth.evictions.values())
    return result
