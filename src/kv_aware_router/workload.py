"""Synthetic chat traffic with realistic prefix structure.

Routing policies can only be compared on traffic that actually contains reuse,
and the shape of that reuse is what decides the winner. Two sources of sharing
matter and they behave differently:

  within a session   turn k repeats everything from turns 1..k-1, so the shared
                     prefix grows as the conversation goes on
  across sessions    unrelated conversations share a system prompt, so there is
                     a short common head even between strangers

A workload with only the first kind flatters session-affinity schemes like
consistent hashing. A workload with only the second flatters nothing. Real chat
traffic has both, so this generates both.

Requests interleave: turns of one session are separated by think time, during
which other sessions arrive. That interleaving is what makes routing hard --
with one conversation at a time every policy looks identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .radix import Tokens


@dataclass(frozen=True, slots=True)
class Request:
    session_id: int
    turn: int
    arrival_s: float
    tokens: Tokens

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    n_sessions: int = 200
    turns_per_session: int = 4
    n_system_prompts: int = 3
    system_tokens: int = 512
    user_tokens: int = 48
    reply_tokens: int = 160
    session_arrival_rate: float = 2.0    # new conversations per second
    think_time_s: float = 20.0           # mean gap between turns
    seed: int = 0


def generate(spec: WorkloadSpec = WorkloadSpec()) -> list[Request]:
    """Build an arrival-ordered list of requests."""
    rng = np.random.default_rng(spec.seed)

    # Distinct system prompts occupy the low token ids so unrelated sessions
    # sharing one produce a genuinely identical head.
    system_prompts = [
        tuple(range(i * spec.system_tokens, (i + 1) * spec.system_tokens))
        for i in range(spec.n_system_prompts)
    ]
    next_token_id = spec.n_system_prompts * spec.system_tokens

    requests: list[Request] = []
    session_start = 0.0
    for session_id in range(spec.n_sessions):
        session_start += rng.exponential(1.0 / spec.session_arrival_rate)
        prompt = system_prompts[rng.integers(spec.n_system_prompts)]

        history: list[int] = list(prompt)
        at = session_start
        for turn in range(spec.turns_per_session):
            user = tuple(range(next_token_id, next_token_id + spec.user_tokens))
            next_token_id += spec.user_tokens
            history.extend(user)

            requests.append(
                Request(session_id, turn, at, tuple(history))
            )

            # The model's reply becomes part of the next turn's prompt, which is
            # why the shared prefix grows rather than staying constant.
            reply = tuple(range(next_token_id, next_token_id + spec.reply_tokens))
            next_token_id += spec.reply_tokens
            history.extend(reply)
            at += rng.exponential(spec.think_time_s)

    requests.sort(key=lambda r: r.arrival_s)
    return requests


def reuse_ceiling(requests: list[Request], match_unit: int = 16) -> float:
    """Best possible prefix reuse: every request routed to a perfect oracle.

    The number a policy is really competing against. Computed by giving a single
    replica infinite capacity, so any prefix ever seen is still resident.
    """
    from .radix import PrefixTree

    tree = PrefixTree(prefix_match_unit=match_unit)
    cached = 0
    total = 0
    for i, req in enumerate(requests):
        cached += tree.match(req.tokens).get("oracle", 0)
        total += req.n_tokens
        tree.insert(req.tokens, "oracle", now=float(i))
    return cached / total if total else 0.0
