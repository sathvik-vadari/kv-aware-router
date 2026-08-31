#!/usr/bin/env python3
"""The two conditions a stateless session hash cannot answer.

Everything measured so far used uniform sessions on a fixed fleet, which is the
best case for consistent hashing: assign by identity and never think again.
Production is neither.

Q1: how does the ranking change when session weight is skewed, so a hash can
    land several heavy conversations on one replica with no way to rebalance?
Q2: how does it change when replicas join and leave, which reshuffles a hash
    ring and invalidates affinity wholesale?
Q3: does a freshly added replica actually get used?

Usage:
    uv run python experiments/05_skew_and_churn.py
"""

from __future__ import annotations

import json
from pathlib import Path

from kv_aware_router.policies import POLICIES, make_policy
from kv_aware_router.simulate import ChurnEvent, simulate
from kv_aware_router.workload import WorkloadSpec, generate

REPLICAS = ["a", "b", "c", "d"]
CAPACITY = 8_000
LOAD = 3.0
OUT = Path(__file__).resolve().parent.parent / "results" / "05_skew_and_churn.json"


def summarise(result, new_replica: str | None = None, after_s: float | None = None) -> dict:
    # after_s must be in *simulated* time, i.e. already divided by load_scale.
    # Outcomes carry scaled arrival times; comparing them against an unscaled
    # threshold silently selects an empty slice and reports 0.0%.
    row = {
        "policy": result.policy,
        "reuse": round(result.reuse_fraction, 4),
        "ttft_p50_ms": round(result.ttft(50) * 1000, 1),
        "ttft_p99_ms": round(result.ttft(99) * 1000, 1),
        "load_cv": round(result.load_cv, 4),
    }
    if new_replica and after_s is not None:
        later = [o for o in result.outcomes if o.arrival_s >= after_s]
        on_new = [o for o in later if o.replica == new_replica]
        row["post_scaleup_share_on_new_replica"] = (
            round(len(on_new) / len(later), 4) if later else 0.0
        )
    return row


def main() -> None:
    out: dict = {"replicas": REPLICAS, "capacity_tokens": CAPACITY, "load_scale": LOAD}

    # Q1 -----------------------------------------------------------------
    print(f"Q1  load skew (fleet fixed, load {LOAD:g}x, {CAPACITY // 1000}k tokens/replica)")
    skew_rows = []
    for skew in [0.0, 1.0, 1.5]:
        requests = generate(WorkloadSpec(n_sessions=250, turns_per_session=4,
                                         turn_skew=skew, seed=5))
        heaviest = {}
        for r in requests:
            heaviest[r.session_id] = heaviest.get(r.session_id, 0) + 1
        counts = sorted(heaviest.values(), reverse=True)
        top_share = sum(counts[: max(1, len(counts) // 10)]) / sum(counts)
        print(f"\n  turn_skew {skew:<4} ({len(requests)} requests, "
              f"top 10% of sessions carry {top_share:.0%})")
        print(f"  {'policy':22s}{'reuse':>8}{'TTFT p99':>11}{'load CV':>10}")
        for name in POLICIES:
            r = simulate(requests, make_policy(name), REPLICAS,
                         capacity_tokens=CAPACITY, load_scale=LOAD)
            row = summarise(r) | {"turn_skew": skew}
            skew_rows.append(row)
            print(f"  {name:22s}{row['reuse']:>7.1%}{row['ttft_p99_ms']:>10.0f}m{row['load_cv']:>10.3f}")
    out["skew"] = skew_rows

    # Q2 and Q3 ----------------------------------------------------------
    requests = generate(WorkloadSpec(n_sessions=250, turns_per_session=4,
                                     turn_skew=1.0, seed=5))
    span = max(r.arrival_s for r in requests)
    scenarios = {
        "stable fleet": None,
        "scale down (lose 1 of 4)": [ChurnEvent(span * 0.4, "remove", "d")],
        "scale up (add 2)": [ChurnEvent(span * 0.35, "add", "e"),
                             ChurnEvent(span * 0.35, "add", "f")],
    }
    print(f"\nQ2/Q3  replica churn (skewed sessions, load {LOAD:g}x)")
    churn_rows = []
    for label, events in scenarios.items():
        print(f"\n  {label}")
        header = f"  {'policy':22s}{'reuse':>8}{'TTFT p99':>11}{'load CV':>10}"
        if events and events[0].action == "add":
            fair = 1.0 / (len(REPLICAS) + 2)
            header += f"{'share of new (fair=' + format(fair, '.1%') + ')':>26}"
        print(header)
        for name in POLICIES:
            r = simulate(requests, make_policy(name), REPLICAS,
                         capacity_tokens=CAPACITY, load_scale=LOAD, churn=events)
            new_replica = "e" if events and events[0].action == "add" else None
            after = (span * 0.35) / LOAD if new_replica else None
            row = summarise(r, new_replica, after) | {"scenario": label}
            churn_rows.append(row)
            line = (f"  {name:22s}{row['reuse']:>7.1%}{row['ttft_p99_ms']:>10.0f}m"
                    f"{row['load_cv']:>10.3f}")
            if "post_scaleup_share_on_new_replica" in row:
                line += f"{row['post_scaleup_share_on_new_replica']:>25.1%}"
            print(line)
    out["churn"] = churn_rows

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote results/{OUT.name}")


if __name__ == "__main__":
    main()
