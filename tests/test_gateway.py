"""The gateway, against fake backends. No ports, no network."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from kv_aware_router.gateway import Config, create_app

BACKENDS = {"a": "http://a.local", "b": "http://b.local", "c": "http://c.local"}


def fake_backends(fail: set[str] | None = None):
    """Every backend echoes which replica served the request."""
    fail = fail or set()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        replica = request.url.host.split(".")[0]
        seen.append(replica)
        if replica in fail:
            return httpx.Response(500, json={"error": "boom"})
        body = json.loads(request.content)
        if body.get("stream"):
            payload = (
                f'data: {{"choices":[{{"delta":{{"content":"{replica}"}}}}]}}\n\n'
                "data: [DONE]\n\n"
            )

            async def sse():
                # must be an unread async stream: a Response built with text=
                # is already consumed, and client.stream() then refuses it
                yield payload.encode()

            return httpx.Response(200, content=sse(),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": replica}}]
        })

    return httpx.MockTransport(handler), seen


def make_client(policy: str = "cost_model", fail: set[str] | None = None, **kw):
    transport, seen = fake_backends(fail)
    app = create_app(
        Config(backends=BACKENDS, policy=policy, **kw),
        client=httpx.AsyncClient(transport=transport),
    )
    return TestClient(app), seen


def chat(messages, **kw):
    return {"model": "test", "messages": messages, **kw}


SYSTEM = {"role": "system", "content": "You are a helpful assistant. " * 60}


def test_health_lists_the_fleet():
    client, _ = make_client()
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["backends"]) == set(BACKENDS)


def test_a_request_reaches_a_backend_and_comes_back():
    client, seen = make_client()
    resp = client.post("/v1/chat/completions", json=chat([SYSTEM, {"role": "user", "content": "hi"}]))
    assert resp.status_code == 200
    assert resp.headers["x-kv-router-replica"] in BACKENDS
    assert seen == [resp.headers["x-kv-router-replica"]]


def test_a_follow_up_turn_returns_to_the_replica_holding_the_cache():
    client, _ = make_client(policy="pure_affinity")
    turn1 = [SYSTEM, {"role": "user", "content": "first question"}]
    first = client.post("/v1/chat/completions", json=chat(turn1))
    turn2 = turn1 + [
        {"role": "assistant", "content": "an answer " * 40},
        {"role": "user", "content": "follow up"},
    ]
    second = client.post("/v1/chat/completions", json=chat(turn2))
    assert second.headers["x-kv-router-replica"] == first.headers["x-kv-router-replica"]
    assert int(second.headers["x-kv-router-cached-tokens"]) > 0


def test_the_first_request_of_a_conversation_has_no_cache():
    client, _ = make_client()
    resp = client.post("/v1/chat/completions", json=chat([{"role": "user", "content": "cold"}]))
    assert resp.headers["x-kv-router-cached-tokens"] == "0"


def test_streaming_passes_the_body_through():
    client, _ = make_client()
    with client.stream("POST", "/v1/chat/completions",
                       json=chat([SYSTEM, {"role": "user", "content": "hi"}], stream=True)) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    assert "data: [DONE]" in text


def test_stats_report_reuse_and_spread():
    client, _ = make_client(policy="pure_affinity")
    turn1 = [SYSTEM, {"role": "user", "content": "q"}]
    client.post("/v1/chat/completions", json=chat(turn1))
    client.post("/v1/chat/completions",
                json=chat(turn1 + [{"role": "assistant", "content": "a " * 40},
                                   {"role": "user", "content": "q2"}]))
    stats = client.get("/stats").json()
    assert stats["requests"] == 2
    assert stats["cached_tokens"] > 0
    assert 0.0 < stats["reuse_fraction"] < 1.0
    assert sum(stats["requests_per_replica"].values()) == 2


def test_in_flight_does_not_leak_after_a_request_finishes():
    client, _ = make_client()
    for _ in range(5):
        client.post("/v1/chat/completions", json=chat([SYSTEM, {"role": "user", "content": "x"}]))
    replicas = client.get("/stats").json()["replicas"]
    assert all(r["in_flight"] == 0 for r in replicas.values())


def test_a_failing_backend_returns_502_without_leaking_state():
    client, _ = make_client(policy="round_robin", fail=set(BACKENDS))
    resp = client.post("/v1/chat/completions", json=chat([{"role": "user", "content": "x"}]))
    assert resp.status_code == 502
    stats = client.get("/stats").json()
    assert stats["failures"] == 1
    assert all(r["in_flight"] == 0 for r in stats["replicas"].values())


def test_session_header_drives_the_hashing_policy():
    client, _ = make_client(policy="consistent_hash")
    picks = set()
    for i in range(20):
        resp = client.post(
            "/v1/chat/completions",
            headers={"X-Session-Id": f"session-{i}"},
            json=chat([SYSTEM, {"role": "user", "content": "q"}]),
        )
        picks.add(resp.headers["x-kv-router-replica"])
    assert len(picks) > 1          # distinct sessions spread across the fleet


def test_the_same_session_header_pins_to_one_replica():
    client, _ = make_client(policy="consistent_hash")
    picks = {
        client.post("/v1/chat/completions", headers={"X-Session-Id": "stable"},
                    json=chat([SYSTEM, {"role": "user", "content": f"q{i}"}])
                    ).headers["x-kv-router-replica"]
        for i in range(10)
    }
    assert len(picks) == 1


def test_a_body_without_messages_is_rejected():
    client, _ = make_client()
    assert client.post("/v1/chat/completions", json={"model": "x"}).status_code == 400


def test_refuses_to_start_with_no_backends():
    with pytest.raises(ValueError, match="no backends"):
        create_app(Config(backends={}))
