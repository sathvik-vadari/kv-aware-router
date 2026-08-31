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

## 2026-08-26 — first measurements, and the thesis is losing

Q: how much prefill does each policy actually avoid, and at what cost in balance?

Built the workload generator (interleaved multi-turn sessions, shared system prompts), the replay
harness, and the first experiment. 55 tests pass. Numbers are in `results/01_policy_comparison.json`.

**Caught my own baseline cheating.** `consistent_hash` scored exactly the oracle ceiling, which was
too good. It was hashing the first 64 tokens — but those are the *system prompt*, identical across
unrelated sessions, so the ring collapsed into "shard by system prompt". Every session using prompt
X pinned to one replica: maximum reuse, ruined balance, and not consistent hashing at all. Now it
takes a session key like a real deployment would. Kept the broken variant as a control, because the
failure is invisible if you only look at the aggregate.

**The honest result: the cost model does not win.** Session-keyed consistent hashing gets 79.8%
reuse to the cost model's 72.4%, and the cost model's balance edge (CV 0.084 vs 0.158) does not buy
back seven points of reuse. Hashing achieves near-oracle reuse while holding no state whatsoever.

Also: round-robin already captures 65.8% of the 80.5% ceiling. Most of the reuse on this workload is
just the 512-token system prompt, which lands on every replica almost immediately. The interesting
part of cache-aware routing is only the session-specific tail.

Why hashing wins right now: unbounded cache, uniform sessions, perfect session locality. That is its
best case, and I built exactly that. The cost model can only pay for itself where hashing is
structurally blind — finite capacity and eviction, load skew, replica churn. None of it simulated.

So finite capacity is next, and cache-state drift stops being a nice-to-have. It is where the thesis
lives or dies.

## 2026-08-26 (later) — finite capacity and drift

Q: does cache-state drift change which routing policy wins?

Built ground-truth replica caches separate from the router's belief, swept capacity from unbounded
to 8k tokens/replica, and modelled two drift causes. 64 tests pass.

**Zero drift on the first run, and that was the finding.** If the router sees every request and
knows the true capacity, its eviction model reproduces the replica exactly — same insertion order,
same LRU, same evictions. Drift is not a property of having a belief, it is a consequence of
specific failures. Naming them turned out to be most of the work:

  wrong capacity assumption      4.5% drift at 2x, 6.0% at 4x
  a second router on the fleet   3.5% drift
  both                           7.7% drift, 23.6% of requests mispredicted

Real capacity is not knowable to a fixed number anyway — it moves with model size,
gpu_memory_utilization, fragmentation and batch size. So the 2x case is the realistic one, not the
pathological one.

**Drift hurts sticky policies most, exactly as predicted.** Consistent hashing mispredicts on 40.5%
of requests vs the cost model's 23.6%: it keeps routing to an assigned replica that evicted the
prefix, and has no way to find out. That mechanism is real and now measured.

**But it still wins on reuse — 65.8% to 62.9%.** The thesis has not recovered.

The thing I have to be careful not to paper over: the metric is biased against my own policy. The
cost model optimises queue delay + prefill; I am measuring only prefill. It is deliberately trading
reuse for balance, and it does buy real balance (CV 0.063 vs 0.158, a 2.5x tighter spread) — but a
prefill-only metric cannot see that. Latency is where it would show.

So: neither proven nor disproven, and switching metrics now without saying so would be cheating.
Next build is a service-time model so TTFT and p99 become measurable. That is the experiment that
settles it.

## 2026-08-28 — latency, and the verdict

Q: does trading cache reuse for load balance pay off in tail latency?

Built the discrete-event simulator: prefill serialised per replica, decode overlapping, decode
slowing prefill. Swept load 0.5x to 6x. 72 tests pass.

**Yes, and the earlier metric was hiding it.** On prefill work alone the cost model looked mediocre.
On TTFT p99 at 6x load it is 158ms against consistent hashing's 282ms — 44% better — for four points
less reuse.

**The headline finding is not the one I expected.** Round-robin has the best tail at every load above
1x, despite doing the most redundant prefill of any policy. Tail latency is set by the worst-loaded
replica, not by average work, so balance beats reuse. The metric the field optimises for KV-aware
routing is hit rate, and hit rate is not what governs the tail.

**Pure affinity is actively harmful**: 382ms p99 at 6x against round-robin's 152ms. It keeps the warm
replica warm by overloading it.

The cost model is the only policy near-best on both axes — within 4% of round-robin's tail with five
points more reuse. That is a Pareto argument, not a win, and I should keep saying it that way.

Where I overstated the thesis: I claimed the policy would slide from affinity to balance across the
load curve. Load CV does fall (0.127 to 0.047), so it rebalances — but reuse barely moves (67.0% to
66.5%). It redistributes among near-equivalent cache options rather than abandoning the cache. Right
mechanism, wrong story.

Next: load skew and replica churn, the two conditions where consistent hashing has no answer at all.

## 2026-08-29 — the gateway

Q: does this run against real endpoints, or is it only a simulator?

Built the OpenAI-compatible gateway and a pluggable tokenizer. 84 tests, all against
httpx.MockTransport backends so there are no ports and no network.

Two bugs worth keeping:

FastAPI lifespan does not run unless TestClient is used as a context manager, so an injected client
was invisible to the handler. Fixed on the design side rather than the test side -- an injected
client should not depend on lifespan at all -- which is the better shape anyway.

httpx.MockTransport builds an already-read response body, so client.stream() refuses it with
StreamConsumed. The mock has to return an unread async stream.

The thing I was most careful about: completing the dispatch on every exit path, including the client
hanging up mid-stream. Miss that and in_flight only ever goes up, and the router gradually convinces
itself the whole fleet is saturated. Nothing would error -- it would just route worse and worse.

Still simulated where it counts: no run against real vLLM yet, so every latency number is from a
service model I wrote.

## 2026-08-29 (later) — real backends

Q: does any of this hold against a real engine rather than my service model?

Three mlx_lm servers on the M4, real prompt cache, real prefill, driven through the actual gateway.

Checked the premise first, since everything depends on it: the backend genuinely reuses prefixes
across requests. Warm system prompt 285ms TTFT, cold 547ms, repeat 172ms. Real and large.

**cost_model gets pure_affinity's median with a better tail** — p50 within 9ms, p90 better by 210ms,
p99 better by 441ms — and beats consistent_hash on every percentile. That is the thesis, measured.

**The simulation's headline finding did not survive.** Experiment 03 said balance beats reuse for
tail latency, with round-robin winning p99. On hardware round-robin has the worst p90. The simulated
conclusion turned out to be sensitive to service-model parameters I guessed -- mainly the ratio of
prefill cost to decode cost. That is the strongest argument yet for spending the GPU budget.

The caveat I have to keep loud: the three replicas share one GPU. Spreading load across them adds no
parallelism, it interleaves work on the same silicon. That biases against balance and towards
affinity, because the thing balance buys does not exist here. So this settles the reuse axis and
cannot settle the balance-versus-affinity tradeoff.

Also hit a real-world trap: port 8001 already had one of my work services on it, and curl reached
that instead of the backend. Moved to 9101-9103.

## 2026-08-30 — correcting the prior-art claim

Searched the literature properly instead of assuming. The area is thoroughly explored and I had told
Sathvik otherwise.

Preble (arXiv:2407.00023) is the direct precursor and states our exact thesis -- "the fundamental
tension between cache locality and load balancing." DualMap (arXiv:2602.06502) is titled almost
literally what this project does. TokenLake, DeepServe, llm-d, Dynamo, GKE Inference Gateway all
cover the ground.

Worse for my claim: cache-state drift is not an open question. llm-d's precise prefix-cache-aware
routing has the router subscribe to KV block allocation and eviction events from each engine and
keep a real-time map. Even the mitigation I would have proposed -- discount the score when state is
stale -- is written down as standard practice.

So the drift work here measures the cost of a known problem and decomposes it by cause. That is
worth something, but it is not a discovery, and the README now says so.

Project stands as a portfolio piece and a way to have learned this area by rebuilding it. It is not
Paper 3.

## 2026-08-31 — skew and churn

Q: does the ranking change in the two conditions a stateless session hash cannot answer?

**Yes, and the thesis finally wins outright.** At turn_skew 1.5 -- top 10% of sessions carrying half
the requests, which is what chat traffic actually looks like -- cost_model takes both best reuse
(69.9% vs consistent_hash 68.2%) and best tail (853ms vs 895ms). Every earlier loss was on uniform
sessions, which is precisely where assigning by identity is already optimal. I had built consistent
hashing's best case and then been surprised it won.

**pure_affinity gives a newly added replica literally zero requests.** Scale up under load and the
new GPUs sit idle forever, because it only ever picks the longest cached prefix and a cold replica
has none. Clearest argument in the whole repo against naive cache-chasing.

**But the cost model has the same disease, just at a higher threshold.** Swept load on a lighter
workload: 0.0% of traffic to a new replica up to 10x, 4.5% at 16x, 17.0% at 24x. Greedy per-request
minimisation never pays the one-off cost of warming a cold replica -- each request really is faster
on a warm one. Correct per request, wrong operationally.

That is exactly why Preble's scheduler is exploitation *and exploration*. Rebuilding the naive
version is how the design choice became obvious, which is the best argument for having built this.

Measurement bug caught: the post-scale-up window was compared in unscaled time against outcomes
carrying scaled arrival times, so the slice was empty and every policy reported 0.0%. Would have
been easy to write up as a finding. Always sanity-check a metric against a policy whose behaviour
you already know -- round_robin reporting 0% share was the tell.
