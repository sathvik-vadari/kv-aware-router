"""An OpenAI-compatible gateway that routes to whichever replica holds the cache.

Put it in front of any set of OpenAI-compatible endpoints -- vLLM, SGLang,
anything speaking /v1/chat/completions -- and it picks the backend, forwards the
request unchanged, and streams the response straight back.

    KV_ROUTER_BACKENDS="a=http://localhost:8001,b=http://localhost:8002" \\
    KV_ROUTER_POLICY=cost_model \\
    uv run uvicorn kv_aware_router.gateway:app --port 8080

Session identity comes from the `X-Session-Id` header, falling back to the
OpenAI `user` field. Only the hashing policies need it; the cache-aware ones
work from the tokens alone, which is the point -- they do not need the caller
to tell them anything.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .fleet import Fleet
from .policies import make_policy
from .radix import DEFAULT_MATCH_UNIT
from .tokenize import ByteTokenizer, HFTokenizer, Tokenizer, render_messages


@dataclass(slots=True)
class Stats:
    requests: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    per_replica: dict[str, int] = field(default_factory=dict)

    @property
    def reuse_fraction(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


@dataclass(slots=True)
class Config:
    backends: dict[str, str]
    policy: str = "cost_model"
    tokenizer: str = "byte"
    match_unit: int = DEFAULT_MATCH_UNIT
    capacity_tokens: int | None = None
    timeout_s: float = 300.0

    @classmethod
    def from_env(cls) -> "Config":
        raw = os.getenv("KV_ROUTER_BACKENDS", "")
        backends: dict[str, str] = {}
        for entry in filter(None, (e.strip() for e in raw.split(","))):
            if "=" not in entry:
                raise ValueError(
                    f"backend {entry!r} must look like 'name=http://host:port'"
                )
            name, url = entry.split("=", 1)
            backends[name.strip()] = url.strip().rstrip("/")
        capacity = os.getenv("KV_ROUTER_CAPACITY_TOKENS")
        return cls(
            backends=backends,
            policy=os.getenv("KV_ROUTER_POLICY", "cost_model"),
            tokenizer=os.getenv("KV_ROUTER_TOKENIZER", "byte"),
            match_unit=int(os.getenv("KV_ROUTER_MATCH_UNIT", DEFAULT_MATCH_UNIT)),
            capacity_tokens=int(capacity) if capacity else None,
        )


def _build_tokenizer(name: str) -> Tokenizer:
    return ByteTokenizer() if name == "byte" else HFTokenizer(name)


def create_app(config: Config, client: httpx.AsyncClient | None = None) -> FastAPI:
    if not config.backends:
        raise ValueError("no backends configured; set KV_ROUTER_BACKENDS")

    replica_ids = list(config.backends)
    fleet = Fleet(
        replica_ids,
        prefix_match_unit=config.match_unit,
        capacity_tokens=(
            {r: config.capacity_tokens for r in replica_ids}
            if config.capacity_tokens
            else None
        ),
    )
    policy = make_policy(config.policy)
    tokenizer = _build_tokenizer(config.tokenizer)
    stats = Stats(per_replica={r: 0 for r in replica_ids})
    owns_client = client is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Only build a client if one was not injected. An injected client must
        # work without lifespan running, so tests can drive the app directly.
        if getattr(app.state, "client", None) is None:
            app.state.client = httpx.AsyncClient(timeout=config.timeout_s)
        yield
        if owns_client and app.state.client is not None:
            await app.state.client.aclose()

    app = FastAPI(title="kv-aware-router", lifespan=lifespan)
    app.state.client = client
    app.state.fleet = fleet
    app.state.stats = stats
    app.state.config = config

    import time

    def _now() -> float:
        return time.monotonic()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "backends": replica_ids, "policy": policy.name}

    @app.get("/stats")
    async def get_stats() -> dict:
        return {
            "policy": policy.name,
            "tokenizer": config.tokenizer,
            "match_unit": config.match_unit,
            "requests": stats.requests,
            "failures": stats.failures,
            "prompt_tokens": stats.prompt_tokens,
            "cached_tokens": stats.cached_tokens,
            "reuse_fraction": round(stats.reuse_fraction, 4),
            "requests_per_replica": dict(stats.per_replica),
            "replicas": {
                rid: {
                    "in_flight": state.in_flight,
                    "pending_prefill_tokens": state.pending_prefill_tokens,
                    "believed_tokens_cached": fleet.tree.tokens_used.get(rid, 0),
                }
                for rid, state in fleet.replicas.items()
            },
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(400, "body must contain a 'messages' array")

        session_key = request.headers.get("X-Session-Id") or body.get("user")
        tokens = tuple(tokenizer.encode(render_messages(messages)))

        replica = policy.choose(fleet, tokens, session_key=session_key)
        cached = fleet.cached_tokens(tokens).get(replica, 0)

        stats.requests += 1
        stats.prompt_tokens += len(tokens)
        stats.cached_tokens += cached
        stats.per_replica[replica] += 1

        dispatch = fleet.dispatch(replica, tokens, now=_now())
        url = f"{config.backends[replica]}/v1/chat/completions"
        headers = {
            "content-type": "application/json",
            # so the caller can see where it went, and so a second hop can too
            "x-kv-router-replica": replica,
        }
        http: httpx.AsyncClient = app.state.client

        if not body.get("stream"):
            try:
                upstream = await http.post(url, json=body, headers=headers)
                upstream.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                stats.failures += 1
                fleet.complete(dispatch, now=_now())
                raise HTTPException(502, f"backend {replica} failed: {exc}") from exc
            fleet.complete(dispatch, now=_now())
            return JSONResponse(
                upstream.json(),
                headers={"x-kv-router-replica": replica,
                         "x-kv-router-cached-tokens": str(cached)},
            )

        async def stream():
            # The dispatch must be completed on every exit path, including the
            # client hanging up mid-stream, or in_flight leaks upward forever
            # and the router slowly convinces itself the fleet is saturated.
            try:
                async with http.stream("POST", url, json=body, headers=headers) as up:
                    up.raise_for_status()
                    async for chunk in up.aiter_raw():
                        yield chunk
            except Exception:  # noqa: BLE001
                stats.failures += 1
                raise
            finally:
                fleet.complete(dispatch, now=_now())

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"x-kv-router-replica": replica,
                     "x-kv-router-cached-tokens": str(cached)},
        )

    return app


app = create_app(Config.from_env()) if os.getenv("KV_ROUTER_BACKENDS") else None
