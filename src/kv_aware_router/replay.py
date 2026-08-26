"""Replay a workload through a policy and measure what it achieved.

Concurrency model: exactly `concurrency` requests are in flight at any moment.
Before dispatching, the oldest in-flight request is completed if the window is
full. This is crude on purpose -- it creates load for the load-aware policies to
react to without inventing service times for prefill and decode, which would put
made-up hardware numbers underneath every result.

What it therefore measures honestly: how much prefill each policy avoids, and how
evenly it spreads work. What it cannot measure: latency. TTFT needs a service
model, and inventing one here would make the numbers look more authoritative than
they are.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .fleet import Dispatch, Fleet
from .policies import Policy
from .workload import Request


@dataclass(slots=True)
class ReplayResult:
    policy: str
    concurrency: int
    n_requests: int
    total_prompt_tokens: int
    cached_tokens: int
    requests_per_replica: dict[str, int] = field(default_factory=dict)

    @property
    def reuse_fraction(self) -> float:
        """Share of prompt tokens served from cache instead of re-prefilled."""
        return self.cached_tokens / self.total_prompt_tokens if self.total_prompt_tokens else 0.0

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
    capacity_tokens: dict[str, int] | None = None,
) -> ReplayResult:
    fleet = Fleet(
        replica_ids,
        prefix_match_unit=match_unit,
        capacity_tokens=capacity_tokens,
    )
    inflight: deque[Dispatch] = deque()
    result = ReplayResult(
        policy=policy.name,
        concurrency=concurrency,
        n_requests=len(requests),
        total_prompt_tokens=0,
        cached_tokens=0,
        requests_per_replica={rid: 0 for rid in replica_ids},
    )

    for req in requests:
        while len(inflight) >= concurrency:
            fleet.complete(inflight.popleft(), now=req.arrival_s)

        replica = policy.choose(fleet, req.tokens)

        # Measure the hit *before* dispatching: dispatch inserts the prefix, at
        # which point it would look cached whether or not it already was.
        hit = fleet.cached_tokens(req.tokens).get(replica, 0)
        result.cached_tokens += hit
        result.total_prompt_tokens += req.n_tokens
        result.requests_per_replica[replica] += 1

        inflight.append(fleet.dispatch(replica, req.tokens, now=req.arrival_s))

    return result
