#!/usr/bin/env python3
"""The router against real inference backends, on real TTFT.

Every latency number in experiments 01-03 comes from a service model written
for this project. This one measures wall-clock time-to-first-token through the
actual gateway, against real engines doing real prefill with a real bounded
prompt cache that really evicts.

Backends: three mlx_lm servers, chosen because they run on a laptop and expose
a bounded prompt cache. The model is small and the hardware is not a datacentre
GPU, so absolute numbers do not transfer -- the *ranking* of policies under
identical traffic is what this is for.

Conversation history uses the model's own replies rather than synthetic filler,
so the shared prefix between turns is genuine.

Setup:
    for p in 9101 9102 9103; do
      uv run python -m mlx_lm server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \\
        --port $p --host 127.0.0.1 --prompt-cache-size 4 --log-level WARNING &
    done

Usage:
    uv run python experiments/04_real_backend.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from kv_aware_router.gateway import Config, create_app

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
BACKENDS = {"a": "http://127.0.0.1:9101", "b": "http://127.0.0.1:9102", "c": "http://127.0.0.1:9103"}
POLICIES = ["round_robin", "least_connections", "consistent_hash", "pure_affinity", "cost_model"]
OUT = Path(__file__).resolve().parent.parent / "results" / "04_real_backend.json"

SYSTEM_PROMPTS = [
    "You are a meticulous systems engineer. Answer precisely and briefly. ",
    "You are a patient teacher. Explain simply and briefly. ",
    "You are a terse code reviewer. Be direct and brief. ",
]
QUESTIONS = [
    "What is a cache?", "Why does batching help?", "What is a queue?",
    "Define latency.", "What is throughput?", "Explain eviction.",
    "What is a hash?", "Why measure percentiles?",
]


def system_prompt(index: int, salt: str) -> str:
    # Salt keeps each policy's run from inheriting the previous run's warm
    # cache, which would make whichever policy ran first look worst.
    return (SYSTEM_PROMPTS[index % len(SYSTEM_PROMPTS)] * 40) + f"[ctx {salt}] "


async def one_turn(client: httpx.AsyncClient, messages: list[dict], session: str,
                   max_tokens: int) -> tuple[float, str, str]:
    """Returns (ttft_seconds, reply_text, replica)."""
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "stream": True, "temperature": 0.0}
    started = time.perf_counter()
    ttft = None
    parts: list[str] = []
    replica = "?"
    async with client.stream("POST", "/v1/chat/completions", json=body,
                             headers={"X-Session-Id": session}) as resp:
        resp.raise_for_status()
        replica = resp.headers.get("x-kv-router-replica", "?")
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - started
                parts.append(delta)
    return ttft or (time.perf_counter() - started), "".join(parts), replica


async def run_session(client: httpx.AsyncClient, sem: asyncio.Semaphore, sid: int,
                      turns: int, salt: str, max_tokens: int, out: list) -> None:
    messages = [{"role": "system", "content": system_prompt(sid, salt)}]
    for turn in range(turns):
        messages.append({"role": "user", "content": QUESTIONS[(sid + turn) % len(QUESTIONS)]})
        async with sem:
            ttft, reply, replica = await one_turn(client, messages, f"s{sid}", max_tokens)
        out.append({"session": sid, "turn": turn, "ttft_s": ttft, "replica": replica})
        messages.append({"role": "assistant", "content": reply})
        await asyncio.sleep(0.05)   # brief think time between turns


async def run_policy(policy: str, sessions: int, turns: int, concurrency: int,
                     max_tokens: int, tokenizer: str) -> dict:
    async with httpx.AsyncClient(timeout=300) as backend_client:
        app = create_app(
            Config(backends=BACKENDS, policy=policy, tokenizer=tokenizer),
            client=backend_client,
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway",
                                     timeout=300) as gw:
            sem = asyncio.Semaphore(concurrency)
            records: list = []
            await asyncio.gather(*(
                run_session(gw, sem, sid, turns, policy, max_tokens, records)
                for sid in range(sessions)
            ))
            stats = (await gw.get("/stats")).json()

    ttfts = sorted(r["ttft_s"] for r in records)
    def pct(p: float) -> float:
        return ttfts[min(len(ttfts) - 1, int(round(p / 100 * (len(ttfts) - 1))))]

    return {
        "policy": policy,
        "requests": len(records),
        "ttft_p50_ms": round(pct(50) * 1000, 1),
        "ttft_p90_ms": round(pct(90) * 1000, 1),
        "ttft_p99_ms": round(pct(99) * 1000, 1),
        "ttft_mean_ms": round(sum(ttfts) / len(ttfts) * 1000, 1),
        "router_reuse_fraction": stats["reuse_fraction"],
        "requests_per_replica": stats["requests_per_replica"],
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--tokenizer", default=MODEL)
    args = ap.parse_args()

    for name, url in BACKENDS.items():
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                (await c.get(f"{url}/v1/models")).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"backend {name} at {url} is not up: {exc}")

    print(f"{len(BACKENDS)} real backends, {args.sessions} sessions x {args.turns} turns, "
          f"concurrency {args.concurrency}\nmodel {MODEL}\n")

    rows = []
    for policy in POLICIES:
        row = await run_policy(policy, args.sessions, args.turns, args.concurrency,
                               args.max_tokens, args.tokenizer)
        rows.append(row)
        print(f"{row['policy']:20s} p50 {row['ttft_p50_ms']:>7.0f}ms  "
              f"p90 {row['ttft_p90_ms']:>7.0f}ms  p99 {row['ttft_p99_ms']:>7.0f}ms  "
              f"reuse {row['router_reuse_fraction']:>6.1%}  {row['requests_per_replica']}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"model": MODEL, "backends": BACKENDS,
                               "config": vars(args), "results": rows}, indent=2))
    print(f"\nwrote results/{OUT.name}")


if __name__ == "__main__":
    asyncio.run(main())
