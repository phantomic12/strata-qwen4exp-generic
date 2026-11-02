# Engram retrain: what it costs, and what to borrow from V4.1-Flash-1017

## Modal pricing (read from modal.com/pricing, 2026-11-02)

| GPU | $/GPU-hr | dense BF16 TFLOPS | $/TFLOPS-hr |
|---|---|---|---|
| B300 | 7.0992 | ~2250 | 0.00316 |
| B200 | 6.2496 | ~2250 | 0.00278 |
| H200 SXM | 4.5396 | 494.7 | 0.00918 |
| **H100 SXM5** | **3.9492** | **494.7** | **0.00798** |
| RTX PRO 6000 | 3.0312 | ~210 | 0.01443 |
| **A100 80 GB** | **2.4984** | **312.0** | **0.00801** |
| A100 40 GB | 2.0988 | 312.0 | 0.00673 |
| **L40S** | **1.9512** | **366.0** | **0.00533** |
| A10 | 1.1016 | 125 | 0.00881 |
| L4 | 0.7992 | 121 | 0.00660 |
| T4 | 0.5904 | 65 | 0.00908 |

Also: CPU $0.04716/core-hr, memory $0.007992/GiB-hr, volumes $0.09/GiB/month, egress $0.04/GiB.
Starter plan is $0/month with **$30 of free compute monthly** (10 GPU concurrency, 100 containers).

### The A100 vs H100 answer

**H100 costs 1.58× more per hour but delivers 1.59× the dense throughput, so per unit of compute they
are identical** ($0.00798 vs $0.00801 per TFLOPS-hr — a 0.4% difference). Neither is the right choice on
price. Two things actually break the tie:

1. **Wall clock.** For a 300M-token run, H100 is 10.9 h vs A100's 17.4 h. Both cost ~$43 on one GPU.
   H100 wins if you care how long it takes, and being preemptible makes shorter jobs cheaper in practice
   (less time exposed to interruption and restart).
2. **L40S is 33% cheaper per FLOP** ($0.00533). It is the correct pick for a *memory-bound* job, which is
   what an engram finetune largely is — the table gather dominates, not the matmuls. 48 GB is also plenty
   for the table shard plus activations.

Ignoring sparsity claims on NVIDIA datasheets matters here: H100's headline 989 TFLOPS is the 2:4-sparse
number. Dense is 494.7. Using 989 would understate H100 cost by 2×.

## What the retrain actually costs

Whittle's active parameters per token are **3.79 B**, not 45 B — 8 of 256 experts fire. That is what
FLOPs scale with, and it is why this is cheap.

```
total experts   32.2 B
non-expert body  2.8 B
ACTIVE / token   3.79 B
fwd  ≈ 2 × active = 7.58 GFLOP/token
fwd+bwd ≈ 3 × fwd  = 22.7 GFLOP/token     (body frozen but still backpropped)
```

At MFU 0.35, one A100-80:

| tokens | A100-80 | H100 SXM5 | L40S |
|---|---|---|---|
| 10M | 0.6 h / $1.4 | 0.4 h / $0.9 | 0.5 h / $1.0 |
| 100M | 5.8 h / $14.5 | 3.6 h / $7.2 | 3.9 h / $7.6 |
| 300M | 17.4 h / $43.5 | 10.9 h / $21.5 | 11.7 h / $22.8 |
| 1B | 57.8 h / $144.7 | 36.5 h / $72.1 | 39.0 h / $76.1 |

Ranges across MFU 0.25–0.45: 300M tokens is **$34–61 on A100-80, $17–31 on H100, $18–32 on L40S**.

### Data, not FLOPs, is the real budget line

The cost is set by how many tokens it takes to put real mass into a 4.88M-bucket head. Measured on
wikitext-103 (`coverage_curve.py`):

```
   100,000 tokens ->    51,563 distinct bigrams   (0.516/token)
 1,000,000 tokens ->   340,246 distinct bigrams   (0.340/token)
fit: distinct ~ n^0.819
```

Sublinear, as Zipf requires — 10× the data gives only 4.2× the distinct n-grams. Projecting through the
coupon-collector bound `distinct = -ln(1-frac) × buckets`:

| fill target | distinct bigrams | unique tokens (1 pass) |
|---|---|---|
| 50% | 3.38M | **16.5M** |
| 90% | 11.24M | **71.4M** |

More passes over the same data add little: the distinct set is bounded by the corpus, not by tokens seen.
So a serious run needs a **corpus with ~70M+ distinct bigrams**, i.e. roughly 200–400M unique tokens,
1–2 epochs. That lands at **$20–60**.

### Concrete stages

| stage | scope | tokens | cost (H100) | cost (A100-80) |
|---|---|---|---|---|
| 0 | hash + coverage measurement only, CPU | — | $0 | $0 |
| 1 | table values only, frozen body, densify heads 3 & 7 | 30M | $2–4 | $4–7 |
| 2 | + `value_proj`/`wkv` in the loop | 100M | $7–12 | $11–20 |
| 3 | + `key_proj` (re-derive head count / hash) — full retrain | 300M+ | $17–31 | $34–61 |

**The $30/month free compute covers all of stages 1 and 2.** Stage 3 is the only one that needs actual
money, and even that fits inside the original $50 cap. Nothing here requires the GSQ/RCO deferral.

Caveat on stage 3: it is not "more training", it is a different model. Rewriting `key_proj` re-addresses
every trained row, so stages 1–2 results do not carry over.

## Could we rip parts from V4.1-Flash-1017's engram?

First, it is a real and much more developed engram. From `deepseek-ai/DeepSeek-V4.1-Flash-1017`,
`config.json` → `text_config`:

```
engram_layer_ids            [1, 14]          Whittle: [2]
engram_num_embeddings       [384006168, 384016682]   Whittle: 39,040,000
engram_max_ngram_size       4                Whittle: 3
engram_vocab_size           16,000,000       Whittle: 4,880,000  (bucket modulus)
engram_n_heads              8                Whittle: 4
engram_head_dim             256              Whittle: 256
engram_compressed_vocab_size 99,092          Whittle: none
```

Two injection sites instead of one, 3-gram–4-gram instead of 2-gram–3-gram, 197 GB of table (fp8) instead
of Whittle's 10 GB. The modeling code is public: `inference/engram.py` (184 lines) and the `Engram` class
at `inference/model.py:296-365`.

### The parts worth taking

**1. Compressed vocab (the biggest fix).** V4.1 hashes over a *normalized* id space, so " The", "the" and
"THE" collide by design. That directly attacks the bucket-overload problem. Measured on Whittle's real
tokenizer with V4.1's exact normalizer chain (`compress_probe.py`):

```
raw vocab               248,070
compressed vocab        184,009   (74.2%, 1.30x smaller)
bigram overload         12,610:1  ->  6,938:1
real-text merge rate    wiki 2.1%   chat 4.0%
```

Two honest observations. The 1.30× compression ratio is *exactly* V4.1's own (129,280 → 99,092), which
means this ratio is a property of the tokenizer family, not something we tuned. And the real-text effect
is small — only 2.1–4.0% of a corpus's distinct bigrams actually merge. So this shrinks the space and is
free to adopt, but it is **not** a step change on its own.

**2. Prime bucket moduli, disjoint per (ngram size, head).** V4.1 gives every (n-gram size, head) its own
prime-sized range, drawn in order and never reused. Whittle instead uses a single modulus 4,880,000 for
all 8 heads with offsets that are exact multiples of it — which is *why* all four heads of an order share
one address (see `ENGRAM_ANATOMY.md`). Using distinct primes decorrelates the heads for free, at
inference cost only. This is the structural fix for the rank-1 addressing defect.

**3. Signed-sqrt gate.** `Engram` gates the write by a normalized dot product of the residual stream
against a key, then `sigmoid(copysign(|dot|.sqrt, dot))`. A gate is exactly what Whittle lacks: its head 3
is *trained* quiet with no adaptive mechanism, so it cannot recover. V4.1's gate lets a quiet head wake up
per-token. This is the most valuable single idea to steal, and it is ~15 lines.

**4. Two injection sites.** Whittle has one (`ple_layer_ids: [2]`). A second site deepens the engram's
influence, but it doubles table size and is a retrain, not a patch.

**5. fp8 with 32-wide blocks.** `ParallelEngramEmbedding` stores the table as `float8_e4m3fn` with a
per-32-dim scale, dequantizing on lookup — 1 byte/value plus a 2-byte scale per 32, against Whittle's
Q8_0 272 B/row for 256 dims. Comparable density, but fp8 has a wider dynamic range than int8 and no
per-block zero point, which suits weights that span several orders of magnitude. We already know IQ4_NL
beats Q4_0 here for exactly that reason.

### The parts NOT worth taking

- **Head count 8 → 4 and n-gram size 3 → 4.** These are `key_proj` input width and bucket layout. Taking
  them is a full retrain, not a rip.
- **The 197 GB table.** 19.7× Whittle's. It works because DeepSeek streams it; it would obliterate the
  "only a little harder to host than Qwen 35B" goal.
- **The compressed vocab's hash multipliers specifically.** V4.1 derives multipliers from
  `engram_compressed_vocab_size` via a per-layer RNG seeded `10007 * layer_id`. Whittle's are already
  baked into trained weights; adopting V4.1's re-addresses all 39M rows.

### So: yes, rip three things, and one of them is nearly free

Ordered by value per unit of work:

1. **Signed-sqrt gate on the engram write** — ~15 lines, no retrain of the hash, testable in stage 1.
   Directly fixes "head 3 cannot recover" because it adds the adaptive mechanism head 3 lacks.
2. **Prime bucket moduli, disjoint per head** — removes the rank-1 addressing defect. Needs a retrain
   (it re-addresses rows) but no architecture change.
3. **Compressed vocab** — free to adopt, 1.30× smaller space, but measured 2.1–4.0% real effect. Do it
   alongside 2, not on its own.

### A capacity caveat that changes the ceiling

Measured live density, corrected for the within-order address sharing:

| | live rows/head | rows per live value |
|---|---|---|
| Whittle | 0.68 of 4.88M | 39.04M / 18.1M live values = **2.16** |
| V4.1 | 1.0 (not measurable from here) | **8.27** × 2 layers |

Whittle allocates ~2.2 live table values per body parameter; V4.1 ~8.3 per layer. Whittle's table is
therefore *relatively* 3.8× smaller than its body needs, which is a plausible reason the head 3 / head 7
slack exists at all. No amount of finetuning changes a capacity ratio — that is the case for eventually
growing the table, and it is the argument for V4.1's 384M-row sizing rather than its n-gram size.
