#!/usr/bin/env python3
"""Compare every routing policy on identical chat traffic.

Q: how much prefill does each policy avoid, and what does it cost in balance?

Usage:
    uv run python experiments/01_policy_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

from kv_aware_router.policies import POLICIES, ConsistentHash, make_policy
from kv_aware_router.replay import replay
from kv_aware_router.workload import WorkloadSpec, generate, reuse_ceiling

REPLICAS = ["a", "b", "c", "d"]
CONCURRENCY = 8
OUT = Path(__file__).resolve().parent.parent / "results" / "01_policy_comparison.json"


class TokenPrefixHash(ConsistentHash):
    """The mis-specified variant, kept as a cautionary control."""

    name = "consistent_hash_token_prefix"

    def choose(self, fleet, tokens, session_key=None):
        return super().choose(fleet, tokens, session_key=None)


def main() -> None:
    spec = WorkloadSpec(n_sessions=200)
    requests = generate(spec)
    ceiling = reuse_ceiling(requests)

    rows = []
    for name in list(POLICIES) + ["consistent_hash_token_prefix"]:
        policy = TokenPrefixHash() if name.endswith("token_prefix") else make_policy(name)
        result = replay(requests, policy, REPLICAS, concurrency=CONCURRENCY)
        rows.append(
            {
                "policy": name,
                "reuse_fraction": round(result.reuse_fraction, 4),
                "load_cv": round(result.load_cv, 4),
                "prefill_tokens_executed": result.prefill_tokens_executed,
                "requests_per_replica": result.requests_per_replica,
            }
        )

    print(
        f"{len(requests)} requests, {len(REPLICAS)} replicas, concurrency {CONCURRENCY}, "
        f"unbounded cache\noracle reuse ceiling {ceiling:.1%}\n"
    )
    print(f"{'policy':32s}{'reuse':>8}{'load CV':>10}{'prefill tokens':>17}")
    for row in rows:
        print(
            f"{row['policy']:32s}{row['reuse_fraction']:>7.1%}"
            f"{row['load_cv']:>10.3f}{row['prefill_tokens_executed']:>17,}"
        )

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "spec": spec.__dict__ if hasattr(spec, "__dict__") else str(spec),
                "replicas": REPLICAS,
                "concurrency": CONCURRENCY,
                "oracle_reuse_ceiling": round(ceiling, 4),
                "results": rows,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
