#!/usr/bin/env python3
"""Tail latency across the load curve -- the metric the cost model optimises.

Q1: how does TTFT p99 move with offered load for each policy?
Q2: does trading cache reuse for load balance pay off in latency, and where?

Usage:
    uv run python experiments/03_latency_under_load.py
"""

from __future__ import annotations

import json
from pathlib import Path

from kv_aware_router.policies import POLICIES, make_policy
from kv_aware_router.simulate import ServiceModel, simulate
from kv_aware_router.workload import WorkloadSpec, generate

REPLICAS = ["a", "b", "c", "d"]
CAPACITY = 8_000
LOADS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
OUT = Path(__file__).resolve().parent.parent / "results" / "03_latency_under_load.json"


def main() -> None:
    spec = WorkloadSpec(n_sessions=200)
    requests = generate(spec)
    service = ServiceModel()

    print(
        f"{len(requests)} requests, {len(REPLICAS)} replicas, "
        f"{CAPACITY // 1000}k tokens/replica\n"
        f"prefill {1 / service.prefill_s_per_token:,.0f} tok/s, "
        f"decode {1 / service.decode_s_per_token:.0f} tok/s/request, "
        f"{service.output_tokens} output tokens\n"
    )

    rows = []
    print("TTFT p99, milliseconds")
    print(f"{'load':>8}" + "".join(f"{n:>19}" for n in POLICIES))
    for load in LOADS:
        cells = {}
        for name in POLICIES:
            r = simulate(
                requests, make_policy(name), REPLICAS,
                service=service, capacity_tokens=CAPACITY, load_scale=load,
            )
            cells[name] = r
            rows.append({
                "load_scale": load, "policy": name,
                "reuse": round(r.reuse_fraction, 4),
                "ttft_p50_ms": round(r.ttft(50) * 1000, 1),
                "ttft_p99_ms": round(r.ttft(99) * 1000, 1),
                "queue_wait_p99_ms": round(r.queue_wait(99) * 1000, 1),
                "load_cv": round(r.load_cv, 4),
            })
        print(f"{load:>7g}x" + "".join(f"{cells[n].ttft(99) * 1000:>18.0f}" for n in POLICIES))

    print("\nprefix reuse")
    print(f"{'load':>8}" + "".join(f"{n:>19}" for n in POLICIES))
    for load in LOADS:
        picked = {r["policy"]: r for r in rows if r["load_scale"] == load}
        print(f"{load:>7g}x" + "".join(f"{picked[n]['reuse']:>18.1%}" for n in POLICIES))

    print("\nat 4x load: the tradeoff, side by side")
    print(f"{'policy':22s}{'reuse':>8}{'TTFT p99':>11}{'load CV':>10}")
    for r in [r for r in rows if r["load_scale"] == 4.0]:
        print(f"{r['policy']:22s}{r['reuse']:>7.1%}{r['ttft_p99_ms']:>10.0f}m{r['load_cv']:>10.3f}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "replicas": REPLICAS,
        "capacity_tokens": CAPACITY,
        "service_model": service.__dict__ if hasattr(service, "__dict__") else str(service),
        "results": rows,
    }, indent=2, default=str))
    print(f"\nwrote results/{OUT.name}")


if __name__ == "__main__":
    main()
