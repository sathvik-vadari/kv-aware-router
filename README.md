# kv-aware-router

A router that sends each LLM request to the replica that already has its KV cache.

## Problem

You run N replicas of a model behind a load balancer. Turn 2 of a conversation shares a long
prefix with turn 1 — the system prompt plus the entire history. If turn 2 lands on a different
replica, that replica re-prefills the whole history from scratch.

For chat workloads prefill is often the bulk of the compute, so this is not a rounding error. And
every standard balancing policy makes it worse: round-robin and least-connections are explicitly
designed to spread requests evenly, which is exactly wrong when the backends have memory. They are
optimising for a stateless world.

## Solution

Model the fleet's collective KV cache as a radix tree over token sequences, so the router can ask:
*which replica already holds the longest usable prefix of this request?*

Then route on a cost model rather than a heuristic:

```
expected_TTFT(replica) = queue_delay(replica) + prefill_cost(uncached_tokens)
route to argmin
```

This matters because cache affinity and load balance are in direct conflict — pure affinity melts
your hot replicas, pure balance gives you no reuse at all. Most implementations pick a side and
bolt on a threshold. The cost model turns that fight into an optimisation, and it degrades
correctly at both ends without tuning: under light load queue delay is near zero so it behaves as
pure affinity, and under heavy load queue delay dominates so it becomes pure balance.

## Results so far

800 requests, 4 replicas, concurrency 8, **unbounded cache**. Regenerate with
`uv run python experiments/01_policy_comparison.py`.

| policy | prefix reuse | load CV |
|---|---|---|
| oracle ceiling | 80.5% | — |
| `round_robin` | 65.8% | 0.000 |
| `least_connections` | 65.8% | 0.000 |
| `consistent_hash` | 79.8% | 0.158 |
| `pure_affinity` | 80.5% | 0.579 |
| **`cost_model`** | **72.4%** | **0.084** |

Two findings, and the second one is inconvenient.

**Most of the available reuse is the shared system prompt.** Cache-blind round-robin still captures
65.8% of the 80.5% ceiling, because a 512-token system prompt gets cached on every replica almost
immediately. The marginal value of cache-aware routing is the session-specific tail, not the
headline number — so any claim of a large win depends entirely on the ratio of conversation history
to shared prefix in the workload.

**Session-keyed consistent hashing beats the cost model here, on both axes that matter.** 79.8% vs
72.4% reuse, and 0.158 vs 0.084 CV is a balance edge that does not pay for seven points of reuse.
Consistent hashing gets near-oracle reuse while holding no state at all.

That result is honest and it is not what this project set out to show. The reason is that these
conditions are consistent hashing's best case: perfect session locality, uniformly weighted
sessions, and an unbounded cache that never evicts anything. The cost model can only earn its
complexity where hashing structurally cannot see the problem:

- **finite capacity** — once replicas evict, hashing keeps routing to a replica that no longer holds
  the prefix, and has no way to find out
- **load skew** — heavy and light sessions hash the same, so there is no rebalancing
- **replica churn** — scaling the fleet reshuffles the ring and invalidates affinity wholesale

## Results with a finite cache

`uv run python experiments/02_capacity_and_drift.py`. Same traffic, 8k tokens per replica
(about 1 GB of KV for an 8B model), with two drift causes active: the router assuming 2× the real
capacity, and a second router sharing the fleet whose traffic it never sees.

| policy | reuse | drift | mispredicted requests | load CV |
|---|---|---|---|---|
| `round_robin` | 60.7% | 4.3% | 15.2% | 0.000 |
| `least_connections` | 60.7% | 4.3% | 15.2% | 0.000 |
| `consistent_hash` | **65.8%** | 12.2% | 40.5% | 0.158 |
| `pure_affinity` | 65.6% | 11.5% | 38.2% | 0.579 |
| `cost_model` | 62.9% | **7.7%** | **23.6%** | **0.063** |

**Drift is not automatic — it has causes.** The first run of this measured zero drift under every
condition, which was the useful result. A router that sees all traffic and knows the true capacity
reproduces the replica's eviction exactly. Drift comes from specific failures: a wrong capacity
assumption (4.5% at 2×, 6.0% at 4×), unobserved traffic from another router (3.5%), and 7.7% when
both are present.

**Drift punishes the sticky policies hardest.** Consistent hashing mispredicts on 40.5% of requests
against the cost model's 23.6%, because it keeps routing to an assigned replica that has evicted the
prefix and has no mechanism to find out. That is the failure this project was built to catch, and it
is real and measurable.

**But the ranking does not change. Consistent hashing still wins on reuse.**

### Why that is not yet a verdict

The metric is structurally biased against the cost model, and it is worth being explicit about it
rather than quietly switching metrics later.

The cost model optimises `queue_delay + prefill_cost(uncached)`. This measures only the second term.
A policy that ignores queueing will always look better on a queueing-free metric — the cost model is
deliberately giving up reuse to buy balance, and the balance it buys is real (CV 0.063 against
0.158, a 2.5× tighter spread). That advantage can only show up as tail latency, and latency needs a
service-time model this harness does not have.

So the honest position at that point: neither proven nor disproven. Measuring TTFT settles it.

## Results with latency

`uv run python experiments/03_latency_under_load.py`. Same fleet, offered load swept on fixed
traffic. TTFT p99 in milliseconds:

| load | `round_robin` | `least_conn` | `consistent_hash` | `pure_affinity` | `cost_model` |
|---|---|---|---|---|---|
| 0.5× | 74 | 74 | 74 | 77 | **74** |
| 1× | 77 | 77 | 81 | 87 | **77** |
| 2× | **84** | 95 | 98 | 118 | 91 |
| 4× | **104** | 148 | 155 | 205 | 127 |
| 6× | **152** | 256 | 282 | 382 | 158 |

| | reuse | TTFT p99 @ 6× | load CV |
|---|---|---|---|
| `round_robin` | 61.5% | **152 ms** | 0.000 |
| `consistent_hash` | **70.6%** | 282 ms | 0.158 |
| `pure_affinity` | 70.5% | 382 ms | 0.579 |
| `cost_model` | 66.5% | 158 ms | 0.047 |

**Chasing the cache is actively harmful at load.** `pure_affinity` is 2.5× worse on tail latency
than round-robin at 6× load — 382 ms against 152 ms — in exchange for nine points of reuse. It
keeps the warm replica warm by overloading it, and the queue it builds costs far more than the
prefill it saves.

**Load balance beats cache reuse for tail latency, at every load above 1×.** Round-robin does the
most redundant prefill work of any policy and still has the best p99, because the tail is set by the
worst-loaded replica rather than by average work. That is worth sitting with: the metric the field
optimises for KV-aware routing — hit rate — is not the metric that governs the tail.

**The cost model is the only policy that is near-best on both axes.** At 6× load it is within 4% of
round-robin's tail (158 ms vs 152 ms) while getting five points more reuse, and it beats consistent
hashing's tail by 44% for four points less reuse. Neither extreme is close to both.

### Where the thesis was overstated

The claim was that one policy would slide from affinity behaviour under light load to balance
behaviour under heavy load. The adaptation is real but smaller and different in kind than claimed:
load CV falls from 0.127 to 0.047 as load rises, so it does rebalance — but reuse barely moves
(67.0% → 66.5%). It is redistributing among near-equivalent cache options rather than abandoning the
cache. The mechanism works; the story about it was too dramatic.

`cost_model` also does not strictly dominate anything. It sits on the Pareto frontier between the
extremes, which is a weaker and more honest claim than "it wins".

Still unmodelled, and each favours the cost model further: load skew across sessions, replica churn
from scaling the fleet, and heterogeneous replica capacity.

## Results on real backends

`uv run python experiments/04_real_backend.py`. Three `mlx_lm` servers running
Qwen2.5-0.5B-Instruct-4bit with a bounded prompt cache, 24 sessions × 4 turns, concurrency 6,
96 requests per policy. Wall-clock TTFT through the actual gateway.

| policy | TTFT p50 | TTFT p90 | TTFT p99 | reuse |
|---|---|---|---|---|
| `round_robin` | 2023 ms | 2726 ms | 3771 ms | 78.5% |
| `least_connections` | 1835 ms | 2447 ms | **2755 ms** | 78.4% |
| `consistent_hash` | 2016 ms | 2813 ms | 3310 ms | 83.0% |
| `pure_affinity` | **1641 ms** | 2520 ms | 3370 ms | **88.1%** |
| `cost_model` | 1650 ms | **2310 ms** | 2929 ms | 84.6% |

The backend's prefix cache is real and large: a warm system prompt gives 285 ms TTFT against
547 ms cold, and 172 ms on a repeat.

**The cost model gets affinity's median with a better tail.** It is within 9 ms of `pure_affinity`
on p50 while beating it by 210 ms on p90 and 441 ms on p99, and it beats `consistent_hash` on every
percentile. That is the intended behaviour, measured rather than modelled.

**The simulation's headline finding did not survive contact with hardware.** Experiment 03 concluded
that balance beats reuse for tail latency and that round-robin has the best p99. Here round-robin
has the *worst* p90 and among the worst p99. The simulated conclusion was sensitive to service-model
parameters — particularly the ratio of prefill cost to decode cost — that were guessed rather than
measured. Worth stating plainly, because it is the reason the rented-GPU run matters.

### The caveat that limits this

**The three replicas share one GPU.** They are three processes on a single M4, so spreading load
across them adds no real parallelism — it just interleaves work on the same silicon. That biases
against the balance-oriented policies and in favour of affinity, because on this setup the main
thing balance buys (independent compute) does not exist.

So this run establishes that the prefix cache is real, that the gateway exploits it, and that the
cost model behaves as designed on the reuse axis. It cannot settle the balance-versus-affinity
tradeoff. That needs replicas on independent GPUs, which is what the rented session is for.

Also: 96 requests per policy is a small sample, so p99 is close to the maximum and should be read
as indicative. Absolute numbers come from a 0.5B model on a laptop and do not transfer.

## Application

A gateway you put in front of any set of OpenAI-compatible endpoints.

## Status

Working gateway, and three experiments with results. Not yet run against a real inference backend —
every latency number below comes from a service model, not hardware.

- [x] Radix tree over tokens with per-replica residency, block-aligned matching, LRU eviction
- [x] Cost-model router and the baselines to beat (round-robin, least-connections, consistent hash, pure affinity)
- [x] Gateway speaking the OpenAI chat completions API
- [x] Cache-state drift: what happens when a replica evicts something the router still believes it has
- [ ] Evaluation under production arrival traces, not synthetic uniform load

New to prefill, decode, and KV cache mechanics? [CONCEPTS.md](CONCEPTS.md) explains what the
router is exploiting and where the numbers come from.

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

## Running

```bash
uv sync
uv run pytest
```

### The gateway

```bash
KV_ROUTER_BACKENDS="a=http://localhost:8001,b=http://localhost:8002" \
KV_ROUTER_POLICY=cost_model \
KV_ROUTER_TOKENIZER=meta-llama/Llama-3.1-8B-Instruct \
uv run uvicorn kv_aware_router.gateway:app --port 8080
```

Then point any OpenAI client at `http://localhost:8080`. Responses carry
`x-kv-router-replica` and `x-kv-router-cached-tokens`, and `GET /stats` shows live fleet state.

| variable | default | |
|---|---|---|
| `KV_ROUTER_BACKENDS` | — | `name=url` pairs, comma separated. Required. |
| `KV_ROUTER_POLICY` | `cost_model` | any of the five |
| `KV_ROUTER_TOKENIZER` | `byte` | a HuggingFace model id, or `byte` for an offline stand-in |
| `KV_ROUTER_MATCH_UNIT` | `16` | must be a multiple of 16 |
| `KV_ROUTER_CAPACITY_TOKENS` | unset | per-replica KV budget, for the router's eviction model |

**Set the tokenizer to the model you are actually serving.** The default `byte` tokenizer keeps the
gateway runnable with no downloads, but it is not any real model's tokenizer, so match lengths — and
therefore reported hit rates — will not correspond to what the backend does.

### The experiments

```bash
uv run python experiments/01_policy_comparison.py
uv run python experiments/02_capacity_and_drift.py
uv run python experiments/03_latency_under_load.py
```
