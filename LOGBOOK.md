# Logbook

## 2026-08-20 — the prefix tree

Q: can a router know which replica holds the longest usable prefix of a request?

Built the radix tree over token sequences: per-replica residency, block-aligned matching,
tail-first LRU eviction. 10 semantics tests pass.

One real bug, found by the test rather than by reading: edge compression silently hid residency.
A 200-token prefix stored as a single compressed edge returned *no hit* for a 50-token query,
because the walk broke partway along the edge and never reached the node holding the residency
record. The blocks physically exist on that replica. Fix credits replicas resident at the child
with the tokens actually shared along the partial edge.

Worth remembering: the compressed edge stands for a chain of implicit nodes, and residency lives
at the child. Any traversal that stops mid-edge has to look ahead or it under-reports.

Next: the router itself, plus the baselines it has to beat.

## 2026-08-22 — the router

Q: can one policy get cache affinity *and* load balance without a threshold to tune?

Built replica state tracking, the TTFT cost model, and five policies. 40 tests pass.

Answer looks like yes, and the crossover test is the evidence: `a` holds 736 of the 800 tokens,
and the cost model keeps picking `a` until `a`'s queue passes ~736 tokens of pending prefill, then
switches away. Nothing configured that number — it falls out of scoring both concerns in seconds.
Light load → behaves as pure affinity. Heavy load → behaves as load balancing.

Two things worth remembering:

Caught an accounting bug before it landed. `complete()` was recomputing the uncached token count,
but `dispatch()` had already inserted the prefix into the tree, so the recomputation always read
zero and `pending_prefill_tokens` would have grown forever. Fixed by returning a handle from
dispatch that carries the number. Lesson: don't recompute state that an earlier step mutated.

The first policy comparison surprised me — `cost_model` picked the *cold* replica when the warm one
still had turn 1 in flight. That was correct: 750 tokens of queued prefill costs more than 800
tokens of fresh prefill on an idle box. The demo was wrong, not the model. Completing turn 1 first
made it pick the warm replica as expected.

Next: the gateway, so this runs against real endpoints instead of asserted state.
