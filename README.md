# kv-aware-router

Routes each LLM request to the replica that already holds its KV cache, scoring cache affinity and
load balance in the same unit so the tradeoff resolves itself.

## The problem

You run N replicas of a model behind a load balancer. Turn 2 of a conversation shares a long prefix
with turn 1 — the system prompt plus the entire history. Land it on the replica that already has
that history and it prefills only the new tokens; land it anywhere else and it reprocesses
everything. For chat workloads prefill is often the bulk of the compute.

Round-robin and least-connections make this worse *by design*. They exist to spread requests evenly,
which is the right answer for stateless backends and the wrong one for backends with memory.

## The approach

Model the fleet's collective KV cache as a radix tree over token sequences, so the router can ask
which replica holds the longest usable prefix. Then route on a cost model rather than a heuristic:

```
expected_TTFT(replica) = queue_delay(replica) + prefill_cost(uncached_tokens)
route to argmin
```

Cache affinity and load balance are in direct conflict, and most implementations pick a side then
bolt on a threshold (*"use affinity unless the replica is over 70% loaded"*). Because both terms
here are **seconds**, they are directly comparable and the tradeoff resolves itself: on an idle
fleet the queue term vanishes and it behaves as pure affinity; under load the queue term dominates
and it becomes load balancing. Nothing to tune.

The threshold is also per-request rather than global. A request with 5,000 tokens cached has a lot
to lose and will wait a long time; one with 100 tokens cached switches away almost immediately. The
queueing it tolerates is exactly the work the cache saves.

## What I found

Five experiments, [full detail in RESULTS.md](RESULTS.md).

**Naive cache-chasing starves new capacity — completely.** Scale up under load and `pure_affinity`
sends the newly added replicas **0%** of traffic. It only ever picks the longest cached prefix, and
a cold replica has none, so you pay for GPUs that do nothing. Its tail latency is also 2.5× worse
than round-robin at 6× load.

**The cost model has the same disease, just later.** It gives a new replica nothing below ~10× load,
4.5% at 16×, 17% at 24×. Greedy per-request minimisation never pays the one-off cost of warming a
cold replica, because each individual request genuinely is faster on a warm one — correct per
request, wrong operationally. This is exactly why [Preble](https://arxiv.org/abs/2407.00023)'s
scheduler is built around exploitation *and exploration*.

**Hit rate is the wrong headline metric.** Round-robin does the most redundant prefill of any policy
and still had a competitive tail, because tail latency is set by the worst-loaded replica rather
than by average work. Optimising cache hit rate and optimising p99 are not the same objective.

**The cost model only wins once session weight is realistic.** On uniform sessions it lost to plain
session-hashing for four experiments straight. At `turn_skew=1.5` — the top 10% of conversations
carrying half the requests, which is what chat traffic looks like — it takes both best reuse (69.9%
vs 68.2%) and best tail (853 ms vs 895 ms). Uniform sessions are precisely the case where assigning
by identity is already optimal; I had built the baseline's best case and then been surprised it won.

**The simulation disagreed with hardware.** A discrete-event model concluded that balance beats
affinity for tail latency and round-robin wins p99. Run against real `mlx_lm` backends, round-robin
had the *worst* p90. The simulated conclusion was sensitive to a prefill-to-decode cost ratio that
was guessed rather than measured — worth knowing about any routing result published on a simulator,
including the ones here.

## Try it

```bash
uv sync
uv run pytest                                          # 94 tests
uv run python experiments/01_policy_comparison.py      # and 02..05
```

### The gateway

```bash
KV_ROUTER_BACKENDS="a=http://localhost:8001,b=http://localhost:8002" \
KV_ROUTER_POLICY=cost_model \
KV_ROUTER_TOKENIZER=meta-llama/Llama-3.1-8B-Instruct \
uv run uvicorn kv_aware_router.gateway:app --port 8080
```

Point any OpenAI client at `http://localhost:8080`. Responses carry `x-kv-router-replica` and
`x-kv-router-cached-tokens`; `GET /stats` shows live fleet state.

| variable | default | |
|---|---|---|
| `KV_ROUTER_BACKENDS` | — | `name=url` pairs, comma separated. Required. |
| `KV_ROUTER_POLICY` | `cost_model` | `round_robin`, `least_connections`, `consistent_hash`, `pure_affinity`, `cost_model` |
| `KV_ROUTER_TOKENIZER` | `byte` | a HuggingFace model id, or `byte` for an offline stand-in |
| `KV_ROUTER_MATCH_UNIT` | `16` | must be a multiple of 16 |
| `KV_ROUTER_CAPACITY_TOKENS` | unset | per-replica KV budget, for the router's eviction model |

**Set the tokenizer to the model you are actually serving.** The default `byte` tokenizer keeps the
gateway runnable with no downloads, but it is not any real model's tokenizer, so match lengths — and
therefore reported hit rates — will not correspond to what the backend does. A mismatch degrades
quietly rather than failing.

New to prefill, decode and KV cache mechanics? [CONCEPTS.md](CONCEPTS.md) explains what the router
is exploiting and where the numbers come from.

## Design notes

Three rules in `radix.py` are easy to get wrong, and each has a test pinning it down:

**Matches round down to the match unit, which is not the physical block size.** A replica holding
100 tokens can serve 96 at a 16-token unit; the trailing partial block is re-prefilled. Reporting
the unrounded number is the easiest way to overstate a routing policy.

The granularity that matters is the engine's *prefix match unit* — how often it computes
prefix-cache keys — not how it physically stores blocks. vLLM separates the two, and the match unit
can be far finer (their docs give 32 against a 1024-token hybrid-model block). Model this with the
physical block size and the router concludes there is no reuse available when there is plenty.

**Residency is inherited upward.** Holding a 900-token prefix means holding every shorter prefix of
it, because they are physically the same blocks. Edge compression makes this subtle: a long prefix
stored as one compressed edge will report no hit for a short query unless partial edge traversal is
credited explicitly. That bug existed here and the test caught it.

**Eviction is tail-first.** A block cannot be freed while a longer prefix built on top of it is
still resident, so only nodes with no resident descendant are evictable. The result is graceful
degradation — a long cached prefix decays into a shorter one rather than vanishing.

The tree is the router's *belief* about replica state, not ground truth. It drifts whenever a
replica evicts something the router didn't predict. Measuring that drift is a core question here,
not an implementation detail.

## Prior art

This is a well-explored problem and this repo does not claim novelty. It is a from-scratch take
built to understand the area by rebuilding it, and to be measured against the real work rather than
to replace it.

- [**Preble**](https://arxiv.org/abs/2407.00023) is the direct precursor: a two-level distributed
  scheduler targeting, in their words, "the fundamental tension between cache locality and load
  balancing" — the same tension this repo's cost model addresses, solved more thoroughly. 1.5–14.5×
  over SGLang.
- [**DualMap**](https://arxiv.org/pdf/2602.06502) does cache affinity and load balancing together
  with dual hash rings, SLO-aware routing and hotspot rebalancing.
- [**TokenLake**](https://arxiv.org/pdf/2508.17219) and
  [**DeepServe**](https://arxiv.org/pdf/2501.14417) cover segment-level prefix pools and
  locality-aware serverless scheduling.
- Production systems: [llm-d](https://llm-d.ai/blog/kvcache-wins-you-can-see), NVIDIA Dynamo,
  GKE Inference Gateway, SGLang's router, vLLM production-stack, AIBrix.

**On cache-state drift, specifically.** An earlier version of this README called it an open question.
That was wrong. [llm-d ships precise prefix-cache-aware
routing](https://github.com/llm-d/llm-d/blob/main/guides/precise-prefix-cache-aware/README.md) in
which the router subscribes to KV block allocation and eviction events from each engine and keeps a
real-time map — which is the clean answer. The mitigation this repo would have proposed, discounting
a replica's score when its cache state is stale, is also already standard practice. What is measured
here is the *cost* of drift and how it decomposes by cause, not the discovery of it.

## Status

Working gateway, five experiments, 94 tests. Validated against real `mlx_lm` backends; not yet run
against vLLM or on independent GPUs, so latency numbers show a *ranking* under identical traffic,
not transferable figures.

- [x] Radix tree over tokens with per-replica residency, block-aligned matching, LRU eviction
- [x] Cost-model router and four baselines to beat
- [x] Gateway speaking the OpenAI chat completions API
- [x] Cache-state drift, decomposed by cause
- [x] Load skew and replica churn
- [x] Validation against real inference backends
- [ ] Evaluation under production arrival traces, not synthetic arrivals
