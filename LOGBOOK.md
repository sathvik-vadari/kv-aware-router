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
