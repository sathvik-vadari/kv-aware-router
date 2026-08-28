#!/usr/bin/env python3
"""Finite KV cache, and what happens when the router's picture goes stale.

Q1: how much does prefix reuse fall as replica cache capacity shrinks?
Q2: what actually causes the router's belief to drift from reality, and by how
    much does each cause move it?
Q3: does drift change which policy wins?

Usage:
    uv run python experiments/02_capacity_and_drift.py
"""

from __future__ import annotations

import json
from pathlib import Path

from kv_aware_router.policies import POLICIES, make_policy
from kv_aware_router.replay import replay
from kv_aware_router.workload import WorkloadSpec, generate

REPLICAS = ["a", "b", "c", "d"]
CONCURRENCY = 8
CAPACITIES = [None, 64_000, 32_000, 16_000, 8_000]
TIGHT = 8_000
OUT = Path(__file__).resolve().parent.parent / "results" / "02_capacity_and_drift.json"


def main() -> None:
    spec = WorkloadSpec(n_sessions=200)
    requests = generate(spec)
    # a second router sharing the same fleet, whose traffic this router never sees
    background = generate(WorkloadSpec(n_sessions=200, seed=99))

    out: dict = {"replicas": REPLICAS, "concurrency": CONCURRENCY}

    print(f"{len(requests)} requests, {len(REPLICAS)} replicas, concurrency {CONCURRENCY}\n")

    # Q1 -----------------------------------------------------------------
    print("Q1  prefix reuse vs replica cache capacity")
    print(f"{'capacity':>14}" + "".join(f"{n:>19}" for n in POLICIES))
    rows = []
    for cap in CAPACITIES:
        label = "unbounded" if cap is None else f"{cap // 1000}k tokens"
        cells = {}
        for name in POLICIES:
            r = replay(requests, make_policy(name), REPLICAS,
                       concurrency=CONCURRENCY, capacity_tokens=cap)
            cells[name] = round(r.reuse_fraction, 4)
        rows.append({"capacity_tokens": cap, "reuse": cells})
        print(f"{label:>14}" + "".join(f"{cells[n]:>18.1%}" for n in POLICIES))
    out["capacity_sweep"] = rows

    # Q2 -----------------------------------------------------------------
    print(f"\nQ2  drift sources, cost_model at {TIGHT // 1000}k tokens/replica")
    print(f"{'source':>34}{'reuse':>9}{'believed':>10}{'drift':>8}{'mispred':>9}")
    sources = [
        ("none (router models it exactly)", {}),
        ("assumes 2x real capacity", {"belief_capacity_ratio": 2.0}),
        ("assumes 4x real capacity", {"belief_capacity_ratio": 4.0}),
        ("second router sharing fleet", {"background": background}),
        ("both", {"belief_capacity_ratio": 2.0, "background": background}),
    ]
    drift_rows = []
    for label, kw in sources:
        r = replay(requests, make_policy("cost_model"), REPLICAS,
                   concurrency=CONCURRENCY, capacity_tokens=TIGHT, **kw)
        drift_rows.append({
            "source": label,
            "reuse": round(r.reuse_fraction, 4),
            "believed_reuse": round(r.believed_reuse_fraction, 4),
            "drift": round(r.drift_fraction, 4),
            "misprediction_rate": round(r.misprediction_rate, 4),
        })
        print(f"{label:>34}{r.reuse_fraction:>8.1%}{r.believed_reuse_fraction:>10.1%}"
              f"{r.drift_fraction:>8.1%}{r.misprediction_rate:>9.1%}")
    out["drift_sources"] = drift_rows

    # Q3 -----------------------------------------------------------------
    print(f"\nQ3  does drift change the ranking? ({TIGHT // 1000}k tokens/replica)")
    realistic = {"belief_capacity_ratio": 2.0, "background": background}
    ranking = []
    for label, kw in [("no drift", {}), ("realistic drift", realistic)]:
        print(f"\n  {label}")
        print(f"  {'policy':22s}{'reuse':>8}{'drift':>8}{'mispred':>9}{'load CV':>10}")
        for name in POLICIES:
            r = replay(requests, make_policy(name), REPLICAS,
                       concurrency=CONCURRENCY, capacity_tokens=TIGHT, **kw)
            ranking.append({
                "condition": label, "policy": name,
                "reuse": round(r.reuse_fraction, 4),
                "drift": round(r.drift_fraction, 4),
                "misprediction_rate": round(r.misprediction_rate, 4),
                "load_cv": round(r.load_cv, 4),
            })
            print(f"  {name:22s}{r.reuse_fraction:>7.1%}{r.drift_fraction:>8.1%}"
                  f"{r.misprediction_rate:>9.1%}{r.load_cv:>10.3f}")
    out["ranking"] = ranking

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote results/{OUT.name}")


if __name__ == "__main__":
    main()
