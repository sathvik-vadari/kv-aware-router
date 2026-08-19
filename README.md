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

## Application

A gateway you put in front of any set of OpenAI-compatible endpoints.

## Status

Early. The prefix tree is built and tested; the router is not written yet.

- [x] Radix tree over tokens with per-replica residency, block-aligned matching, LRU eviction
- [ ] Cost-model router and the baselines to beat (round-robin, least-connections, consistent hash, pure affinity)
- [ ] Gateway speaking the OpenAI chat completions API
- [ ] Cache-state drift: what happens when a replica evicts something the router still believes it has
- [ ] Evaluation under production arrival traces, not synthetic uniform load

## Design notes

Three rules in `radix.py` are easy to get wrong, and each has a test pinning it down:

**Matches round down to a block boundary.** Engines cache in fixed-size blocks — vLLM defaults to
16 tokens. A replica holding 100 tokens can serve 96 of them; the trailing partial block is
re-prefilled. Reporting the unrounded number is the easiest way to overstate a routing policy.

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
