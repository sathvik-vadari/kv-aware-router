"""Replica state, and the router's view of what each replica is holding and doing.

The prefix tree answers "who has the cache". This adds "who is busy", because
routing on cache alone melts whichever replica happens to hold the popular
prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .radix import DEFAULT_MATCH_UNIT, PrefixTree, Tokens


@dataclass(slots=True)
class ReplicaState:
    """What the router believes about one replica right now."""

    replica_id: str
    in_flight: int = 0
    # Prefill tokens accepted but not yet known to be finished. This is the
    # router's estimate of the queue ahead of a newly arriving request.
    pending_prefill_tokens: int = 0

    def __post_init__(self) -> None:
        if self.in_flight < 0 or self.pending_prefill_tokens < 0:
            raise ValueError("replica state cannot start negative")


@dataclass(slots=True)
class CostModel:
    """Parameters for estimating time-to-first-token.

    Deliberately crude, and the crudeness is the point: every term here is a
    number that can be calibrated against a real backend, and calibrating them
    is a task in itself. What it models:

      * prefill is linear in token count
      * queued prefill work delays a new arrival
      * replicas busy decoding for other requests start your prefill later

    What it ignores, knowingly: the quadratic term in attention, chunked
    prefill interleaving, batching effects, and the fact that decode steps get
    slower as the batch grows. Each of those is a reason a measured TTFT will
    differ from this estimate, which is worth knowing before trusting it.
    """

    prefill_s_per_token: float = 1e-4        # ~10k tokens/s prefill
    decode_interference_s: float = 5e-3      # per in-flight request


@dataclass(slots=True)
class Dispatch:
    """Handle for an in-flight request, returned by `Fleet.dispatch`."""

    replica_id: str
    tokens: Tokens
    uncached_tokens: int


class Fleet:
    """Replica states plus the shared prefix tree, kept in sync by lifecycle calls."""

    def __init__(
        self,
        replica_ids: list[str],
        *,
        prefix_match_unit: int = DEFAULT_MATCH_UNIT,
        capacity_tokens: dict[str, int] | None = None,
        backend: str | None = None,
    ) -> None:
        if not replica_ids:
            raise ValueError("a fleet needs at least one replica")
        self.tree = PrefixTree(
            prefix_match_unit=prefix_match_unit,
            capacity_tokens=capacity_tokens,
            backend=backend,
        )
        self.replicas: dict[str, ReplicaState] = {
            rid: ReplicaState(rid) for rid in replica_ids
        }

    @property
    def replica_ids(self) -> list[str]:
        return list(self.replicas)

    # -- what the router needs to decide --------------------------------

    def cached_tokens(self, tokens: Tokens) -> dict[str, int]:
        """Usable cached prefix length per replica, block-aligned."""
        return self.tree.match(tokens)

    def uncached_tokens(self, tokens: Tokens) -> dict[str, int]:
        """Tokens each replica would have to prefill from scratch."""
        cached = self.tree.match(tokens)
        return {rid: len(tokens) - cached.get(rid, 0) for rid in self.replicas}

    def expected_ttft_s(self, tokens: Tokens, cost: CostModel) -> dict[str, float]:
        """queue_delay + prefill_cost(uncached), per replica."""
        uncached = self.uncached_tokens(tokens)
        out: dict[str, float] = {}
        for rid, state in self.replicas.items():
            queue_delay = (
                state.pending_prefill_tokens * cost.prefill_s_per_token
                + state.in_flight * cost.decode_interference_s
            )
            out[rid] = queue_delay + uncached[rid] * cost.prefill_s_per_token
        return out

    # -- lifecycle -------------------------------------------------------

    def dispatch(self, replica_id: str, tokens: Tokens, now: float) -> "Dispatch":
        """Record that a request was sent to a replica.

        Returns a handle that must be passed back to `complete`. The uncached
        count has to be captured here, because inserting the prefix into the
        tree immediately makes it look cached -- recomputing it at completion
        would always read zero and the queue estimate would never drain.
        """
        state = self.replicas[replica_id]
        uncached = len(tokens) - self.tree.match(tokens).get(replica_id, 0)
        state.in_flight += 1
        state.pending_prefill_tokens += uncached
        # Insert at dispatch rather than completion, deliberately: a second
        # request arriving mid-prefill should still be routed to this replica.
        self.tree.insert(tokens, replica_id, now)
        return Dispatch(replica_id=replica_id, tokens=tokens, uncached_tokens=uncached)

    def complete(self, dispatch: "Dispatch", now: float) -> None:
        """Record that a dispatched request finished."""
        state = self.replicas[dispatch.replica_id]
        state.in_flight = max(0, state.in_flight - 1)
        state.pending_prefill_tokens = max(
            0, state.pending_prefill_tokens - dispatch.uncached_tokens
        )
        self.tree.touch(dispatch.tokens, dispatch.replica_id, now)

    def add_replica(self, replica_id: str) -> None:
        self.replicas.setdefault(replica_id, ReplicaState(replica_id))

    def remove_replica(self, replica_id: str) -> None:
        self.replicas.pop(replica_id, None)
        self.tree.remove_replica(replica_id)
