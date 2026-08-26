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

None of those are simulated yet. Finite capacity is next, which makes cache-state drift the
make-or-break question for the thesis rather than an interesting aside.

## Application

A gateway you put in front of any set of OpenAI-compatible endpoints.

## Status

The routing core is built and tested. Nothing has been measured yet — no gateway, no traffic.

- [x] Radix tree over tokens with per-replica residency, block-aligned matching, LRU eviction
- [x] Cost-model router and the baselines to beat (round-robin, least-connections, consistent hash, pure affinity)
- [ ] Gateway speaking the OpenAI chat completions API
- [ ] Cache-state drift: what happens when a replica evicts something the router still believes it has
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

Cache-aware routing exists — SGLang ships a router, and vLLM's production-stack and AIBrix have
their own. This is a from-scratch take with an explicit cost model, built to be measured against
those baselines rather than to replace them.

## Running

```bash
uv sync
uv run pytest
```
