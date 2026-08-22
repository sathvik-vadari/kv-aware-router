# Concepts

Everything this router does follows from one property of attention. Written down here because
the vocabulary (prefill, decode, TTFT, blocks) only makes sense once you have the mechanic.

## The mechanic

A model predicts the next token from all previous tokens. Attention makes each token look at every
earlier one, and for that lookup every earlier token needs two vectors computed from it: a **Key**
and a **Value**.

The property that matters: **token 5's K and V never change.** They depend on token 5 and what came
before it, never on what comes after.

So when generating token 501 you need K,V for tokens 1–500 — and you computed those while
generating token 500. Recomputing them is pure waste, so you keep them. That store is the
**KV cache**: Key and Value vectors, one pair per token, per layer, per attention head.

## Two phases with opposite bottlenecks

**Prefill** processes the prompt. All its tokens are already known, so they compute **in parallel**
as one large matrix multiply. Prefill is **compute-bound** and its cost scales with prompt length.
It ends with the KV cache populated and the first output token emitted.

**Decode** generates the response one token at a time. It **cannot** be parallelised — token 3
depends on token 2 depends on token 1. Each step processes a single token but must read the entire
model weights out of memory to do it, so decode is **memory-bandwidth-bound** and leaves the
arithmetic units mostly idle.

|                 | Prefill        | Decode                  |
| --------------- | -------------- | ----------------------- |
| Tokens per step | all of them    | exactly one             |
| Parallel        | yes            | no, strictly serial     |
| Bottleneck      | compute        | memory bandwidth        |
| Scales with     | prompt length  | output length × model size |

The serving metrics are just these two phases measured separately:

- **TTFT** (time to first token) = queue wait + prefill. Driven by prompt length.
- **TPOT** / **ITL** (time per output token / inter-token latency) = one decode step. Roughly
  constant, driven by model size and memory bandwidth.
- Total ≈ TTFT + TPOT × output tokens.

## How big the cache gets

Per token: `2 × layers × kv_heads × head_dim × bytes`.

Llama-3-8B (32 layers, 8 KV heads, head_dim 128, fp16) → **128 KB per token**. A 2,000-token
conversation is 256 MB. An 80 GB H100 holding ~16 GB of weights has ~60 GB left for KV, so roughly
200 concurrent conversations of that size before it is full.

That number is why the KV cache is the binding constraint on how many users one GPU serves.

## Blocks, and why 16

Storing each conversation's cache contiguously would mean reserving space for the longest possible
output and fragmenting memory badly. vLLM's **PagedAttention** borrows OS virtual memory: chop the
cache into fixed-size **blocks**, keep a block table mapping logical positions to physical blocks.
No fragmentation — and blocks become *shareable* between requests starting with the same tokens,
which is what makes prefix caching possible at all.

Block size is a compromise between two forces:

- **Smaller** is better for sharing. You can only reuse whole units, so a 100-token shared prefix
  yields 96 reusable tokens at 16, 64 at 64, and **nothing** at 128. Small blocks also waste less
  on the partially-filled final block of each sequence.
- **Larger** is better for the kernel. Blocks are gathered from scattered memory, so smaller blocks
  mean more indirection, worse coalescing, and a longer block table to walk every decode step.

**16 is the quantum because of Tensor Cores.** The fp16 MMA instruction shape on Ampere/Hopper is
`m16n8k16`, and in the second attention matmul the KV sequence axis lands on that 16-wide K
dimension. So the hardware consumes the cache 16 rows at a time: a 16-token block is exactly one
tile, 48 is exactly three, and 25 is 1.5625 — leaving a partial tile at every boundary that the
kernel must mask off, on every block, head, layer and step.

Hence vLLM's FlashAttention backend advertising `MultipleOf(16)`. Note that means **48 and 96 are
legal** — the rule is multiples of 16, not powers of two. The block-table division that powers of
two would speed up happens once per block, while tile alignment costs once per element, so tiling
wins. Backends add their own ceilings: FlashInfer asserts `page_size <= 64` unless it is on
Blackwell with the trtllm-gen decode kernel.

**Match unit ≠ block size.** vLLM separates how often prefix-cache keys are computed
(`prefix_match_unit`) from how blocks are physically stored (`block_size`), and the match unit can
be much finer — their docs give 32 against a 1024-token hybrid-model block. This router models the
match unit, because that is what determines reuse.

## Why the router exists

**Turn 1:** system prompt (500 tokens) + user message (50). Prefill 550, generate 200. That
replica's KV cache now holds 750 entries.

**Turn 2:** the prompt is system prompt (500) + user message (50) + assistant reply (200) + new
message (50) = 800 tokens. The first 750 are identical to what that replica already holds.

- Route back to it → prefill 50 tokens.
- Route anywhere else → prefill all 800 from scratch.

Same request, ~16× the prefill work, decided entirely by where it landed. Round-robin and
least-connections pick the second option most of the time, by design, because they were built for
backends without memory.
