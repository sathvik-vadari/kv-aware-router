# Results

The full record, in the order the experiments happened, including the runs where the cost model
lost. [README](README.md) has the summary; [LOGBOOK](LOGBOOK.md) has the session-by-session diary.

Every experiment is reproducible: `uv run python experiments/0N_*.py`, with raw output in
`results/`.

Read in order, this is also the argument for why the early losses were not the cost model's fault:
experiments 01–03 all ran on uniform sessions with a fixed fleet, which is precisely the condition
under which a stateless session hash is already optimal. Experiment 05 changed that and the ranking
changed with it.

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

## Results under skew and churn

`uv run python experiments/05_skew_and_churn.py`. Everything above used uniform sessions on a fixed
fleet, which is consistent hashing's best case. Production is neither.

**Skewed session weight** — turns per session drawn from a lognormal with the mean held fixed, so a
few conversations run for dozens of turns. Load 3×, 8k tokens/replica:

| policy | reuse @ 0 | reuse @ 1.5 | p99 @ 0 | p99 @ 1.5 |
|---|---|---|---|---|
| `round_robin` | 60.9% | 53.3% | **101 ms** | 811 ms |
| `consistent_hash` | 69.2% | 68.2% | 172 ms | 895 ms |
| `pure_affinity` | 69.2% | 67.7% | 239 ms | 895 ms |
| `cost_model` | 65.0% | **69.9%** | 112 ms | **853 ms** |

At skew 1.5 — the top 10% of sessions carrying half the requests — **the cost model wins both axes
for the first time**, taking best reuse *and* best tail. That is the regime it was designed for, and
it explains the earlier losses: uniform sessions are precisely the case where assigning by identity
is already optimal.

(Comparisons are valid within a skew level, where all policies see identical traffic. Across skew
levels the request counts differ, so the trend down a column is not load-matched.)

**Replica churn** — share of post-scale-up requests reaching two newly added replicas, fair share
16.7%:

| policy | share of new capacity |
|---|---|
| `round_robin` | 16.8% |
| `consistent_hash` | 15.7% |
| `least_connections` | 10.5% |
| `cost_model` | 16.8% |
| `pure_affinity` | **0.0%** |

**`pure_affinity` gives a newly added replica literally nothing.** It only ever picks the longest
cached prefix and a cold replica has none, so you scale up under load and the new GPUs sit idle
forever. That is a serious operational failure and the clearest argument in this repo against naive
cache-chasing.

### A limitation of the cost model, found the same way

The cost model takes its fair share above — but only because that run was loaded enough. Sweeping
load on a lighter workload:

| load | share of new replica |
|---|---|
| 1×–10× | 0.0% |
| 16× | 4.5% |
| 24× | 17.0% |

**Below roughly 16× it starves new capacity too.** Greedy per-request cost minimisation never pays
the one-off price of warming a cold replica, because each individual request genuinely is faster on
a warm one. Correct per request, wrong operationally: you scaled up for a reason and the router will
not use it until things are already bad.

This is exactly why [Preble](https://arxiv.org/abs/2407.00023)'s scheduler is built around
exploitation *and exploration* rather than pure exploitation. Rebuilding the naive version is how
that design choice became obvious.

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
