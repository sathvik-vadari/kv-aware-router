"""Routing policies, including the four baselines the cost model has to beat.

Every policy answers the same question -- given this request's tokens and the
current fleet state, which replica? -- so they can be swapped and measured
against each other on identical traffic.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from .fleet import CostModel, Fleet
from .radix import Tokens


@runtime_checkable
class Policy(Protocol):
    name: str

    def choose(self, fleet: Fleet, tokens: Tokens) -> str: ...


def _tiebreak(fleet: Fleet, candidates: list[str]) -> str:
    """Least loaded, then lexical, so tests are deterministic."""
    return min(candidates, key=lambda r: (fleet.replicas[r].in_flight, r))


class RoundRobin:
    """Cycle through replicas. The default almost everywhere, and cache-blind."""

    name = "round_robin"

    def __init__(self) -> None:
        self._n = 0

    def choose(self, fleet: Fleet, tokens: Tokens) -> str:
        ids = fleet.replica_ids
        pick = ids[self._n % len(ids)]
        self._n += 1
        return pick


class LeastConnections:
    """Fewest in-flight requests. Load-aware, still cache-blind."""

    name = "least_connections"

    def choose(self, fleet: Fleet, tokens: Tokens) -> str:
        return _tiebreak(fleet, fleet.replica_ids)


class ConsistentHash:
    """Hash a session key onto a ring of replicas.

    The cheap way to get affinity: hash something stable about the conversation
    and always send it to the same replica. Approximated here by hashing the
    first `session_key_tokens` tokens, since a conversation's opening tokens
    stay fixed as it grows.

    It gets affinity without ever looking at the cache, which is the weakness:
    it cannot tell that a replica evicted the prefix, cannot react to load at
    all, and sends two conversations sharing a long system prompt to different
    replicas whenever their hashes differ.
    """

    name = "consistent_hash"

    def __init__(self, vnodes: int = 100, session_key_tokens: int = 64) -> None:
        self.vnodes = vnodes
        self.session_key_tokens = session_key_tokens
        self._ring: list[tuple[int, str]] = []
        self._built_for: tuple[str, ...] = ()

    @staticmethod
    def _hash(data: bytes) -> int:
        return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

    def _rebuild(self, ids: list[str]) -> None:
        ring = [
            (self._hash(f"{rid}#{v}".encode()), rid)
            for rid in ids
            for v in range(self.vnodes)
        ]
        ring.sort()
        self._ring = ring
        self._built_for = tuple(ids)

    def choose(self, fleet: Fleet, tokens: Tokens) -> str:
        ids = fleet.replica_ids
        if tuple(ids) != self._built_for:
            self._rebuild(ids)
        key = tokens[: self.session_key_tokens]
        h = self._hash(b",".join(str(t).encode() for t in key))
        for point, rid in self._ring:
            if point >= h:
                return rid
        return self._ring[0][1]


class PureAffinity:
    """Always the longest cached prefix. Maximises hit rate, ignores load."""

    name = "pure_affinity"

    def choose(self, fleet: Fleet, tokens: Tokens) -> str:
        cached = fleet.cached_tokens(tokens)
        if not cached:
            return _tiebreak(fleet, fleet.replica_ids)
        best = max(cached.values())
        return _tiebreak(fleet, [r for r, n in cached.items() if n == best])


class CostModelPolicy:
    """Route to the lowest estimated time-to-first-token.

    The project's thesis. Cache affinity and load balance are in direct
    conflict, and picking a side means being wrong at one end of the load
    curve. Scoring both in the same unit -- seconds -- makes the tradeoff
    resolve itself: under light load queue delay is near zero so this behaves
    as pure affinity, and under heavy load queue delay dominates so it becomes
    load balancing. No threshold to tune.
    """

    name = "cost_model"

    def __init__(self, cost: CostModel | None = None) -> None:
        self.cost = cost or CostModel()

    def choose(self, fleet: Fleet, tokens: Tokens) -> str:
        scores = fleet.expected_ttft_s(tokens, self.cost)
        best = min(scores.values())
        return _tiebreak(fleet, [r for r, s in scores.items() if s == best])


POLICIES: dict[str, type] = {
    "round_robin": RoundRobin,
    "least_connections": LeastConnections,
    "consistent_hash": ConsistentHash,
    "pure_affinity": PureAffinity,
    "cost_model": CostModelPolicy,
}


def make_policy(name: str, **kwargs) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(POLICIES)}")
    return POLICIES[name](**kwargs)
