"""Replay a workload through a policy and measure what it achieved.

Two separate structures are maintained here, and keeping them apart is the
whole point of the experiment:

  ground truth   what each replica physically holds, with finite capacity and
                 its own LRU eviction. The replica evicts when it runs out of
                 room; nobody asks the router's permission.

  belief         what the router thinks each replica holds. Maintained by the
                 router from the requests it has dispatched.

Drift between them is not automatic. If the router sees every request and knows
the true capacity, its eviction model reproduces the replica exactly and belief
stays perfect -- which is what a first run of this showed. Drift has causes, and
naming them is most of the problem:

  wrong capacity        the router does not know how many tokens actually fit.
                        Real capacity moves with model size, gpu_memory_utilization,
                        fragmentation and the concurrent batch, so any fixed
                        assumption is wrong in one direction or the other.
  unobserved traffic    another router, another tenant, or a health check touches
                        the same replicas. Their requests evict things this router
                        still believes are resident.
  eviction mismatch     the replica's real policy is not the router's model of it.
  restarts              a replica comes back empty and the router does not notice.

The first two are modelled here.

Concurrency model: exactly `concurrency` requests in flight, oldest completed
when the window is full. Crude on purpose -- it creates load for the load-aware
policies without inventing prefill and decode service times, which would put
made-up hardware numbers underneath every result. The cost is that latency
cannot be measured here, only work and balance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from .fleet import Dispatch, Fleet
from .policies import Policy
from .radix import PrefixTree
from .workload import Request

BeliefMode = Literal["oracle", "modelled", "blind"]


@dataclass(slots=True)
class ReplayResult:
    policy: str
    concurrency: int
    belief: str
    capacity_tokens: int | None
    n_requests: int
    total_prompt_tokens: int
    # what the replica could actually serve from cache
    cached_tokens: int
    # what the router expected before dispatching
    believed_cached_tokens: int
    # requests where the router expected more cache than existed
    mispredicted_requests: int
    overpredicted_tokens: int
    evictions: int
    requests_per_replica: dict[str, int] = field(default_factory=dict)

    @property
    def reuse_fraction(self) -> float:
        """Share of prompt tokens genuinely served from cache."""
        return self.cached_tokens / self.total_prompt_tokens if self.total_prompt_tokens else 0.0

    @property
    def believed_reuse_fraction(self) -> float:
        """What the router thought it was achieving."""
        return (
            self.believed_cached_tokens / self.total_prompt_tokens
            if self.total_prompt_tokens
            else 0.0
        )

    @property
    def drift_fraction(self) -> float:
        """Share of prompt tokens the router expected from cache but did not get.

        Zero means the router's model of the fleet is accurate. Large means it is
        routing on a picture of the world that no longer exists.
        """
        return (
            self.overpredicted_tokens / self.total_prompt_tokens
            if self.total_prompt_tokens
            else 0.0
        )

    @property
    def misprediction_rate(self) -> float:
        return self.mispredicted_requests / self.n_requests if self.n_requests else 0.0

    @property
    def prefill_tokens_executed(self) -> int:
        return self.total_prompt_tokens - self.cached_tokens

    @property
    def load_cv(self) -> float:
        """Coefficient of variation of per-replica request counts.

        0.0 is perfectly even. Reported alongside reuse because a policy that
        wins on reuse by sending everything to one replica has not won anything.
        """
        counts = list(self.requests_per_replica.values())
        if not counts:
            return 0.0
        mean = sum(counts) / len(counts)
        if mean == 0:
            return 0.0
        var = sum((c - mean) ** 2 for c in counts) / len(counts)
        return var**0.5 / mean


def replay(
    requests: list[Request],
    policy: Policy,
    replica_ids: list[str],
    *,
    concurrency: int = 1,
    match_unit: int = 16,
    capacity_tokens: int | None = None,
    belief: BeliefMode = "modelled",
    belief_capacity_ratio: float = 1.0,
    background: list[Request] | None = None,
) -> ReplayResult:
    """Run `requests` through `policy`.

    capacity_tokens
        Per-replica KV cache size. None means unbounded, in which case nothing
        ever evicts and belief can never drift.

    belief
        How the router models replica memory.
          oracle    perfect information -- belief *is* ground truth. The upper
                    bound, and not achievable in production.
          modelled  the router runs its own eviction model with the same
                    capacity assumption. Right policy, but it cannot see the
                    replica's actual LRU order, so it drifts.
          blind     the router assumes replicas never forget. What a scheme
                    with no cache tracking effectively believes.

    belief_capacity_ratio
        How wrong the router's capacity assumption is, under `modelled`. 2.0
        means it believes replicas hold twice what they do, so it keeps routing
        to prefixes that were evicted. 0.5 means it under-uses a cache that is
        really there.

    background
        Requests that reach the replicas without passing through this router --
        a second router sharing the fleet, or another tenant. They evict, and
        the router never learns.
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

    inflight: deque[Dispatch] = deque()
    result = ReplayResult(
        policy=policy.name,
        concurrency=concurrency,
        belief=belief,
        capacity_tokens=capacity_tokens,
        n_requests=len(requests),
        total_prompt_tokens=0,
        cached_tokens=0,
        believed_cached_tokens=0,
        mispredicted_requests=0,
        overpredicted_tokens=0,
        evictions=0,
        requests_per_replica={rid: 0 for rid in replica_ids},
    )

    # Interleave unobserved traffic by arrival time. It is dispatched round-robin
    # as a second router would, and only ever touches ground truth.
    bg = deque(sorted(background or [], key=lambda r: r.arrival_s))
    bg_next = 0

    for req in requests:
        while bg and bg[0].arrival_s <= req.arrival_s:
            other = bg.popleft()
            truth.insert(other.tokens, replica_ids[bg_next % len(replica_ids)], now=other.arrival_s)
            bg_next += 1

        while len(inflight) >= concurrency:
            fleet.complete(inflight.popleft(), now=req.arrival_s)

        replica = policy.choose(fleet, req.tokens, session_key=str(req.session_id))

        # Measure before dispatching: dispatch inserts the prefix, after which
        # it looks cached whether or not it already was.
        actual = truth.match(req.tokens).get(replica, 0)
        believed = fleet.tree.match(req.tokens).get(replica, 0)

        result.cached_tokens += actual
        result.believed_cached_tokens += believed
        result.total_prompt_tokens += req.n_tokens
        result.requests_per_replica[replica] += 1
        if believed - actual >= match_unit:
            result.mispredicted_requests += 1
            result.overpredicted_tokens += believed - actual

        truth.insert(req.tokens, replica, now=req.arrival_s)
        inflight.append(fleet.dispatch(replica, req.tokens, now=req.arrival_s))

    result.evictions = sum(truth.evictions.values())
    return result
