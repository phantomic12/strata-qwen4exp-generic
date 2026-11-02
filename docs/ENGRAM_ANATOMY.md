# The Whittle engram: what it is, what it does, and what can actually be improved

Measured on `whittleOrg/Whittle-Qwen-3.8-45B-A3B`, table `per_layer_token_embd.weight`,
`[256, 39040000]`, 40 layers, one injection site (`ple_layer_ids: [2]`).

## Geometry

- 39,040,000 rows × 256 dims. 8 heads × 4,880,000 rows each.
- `ngram_size 3`, `heads_per_ngram 4` → **4 bigram heads (0–3) + 4 trigram heads (4–7)**.
- 256 × 8 = 2048 = `n_embd`. The concatenated gather *is* the layer-2 input width.
- Head offsets are a plain partition of the table: `0, 4880000, …, 34160000`.
- Q8_0 in the released file: 272 B/row → 10.62 GB, 22% of the 48 GB file.

## The hash, verified

```
mixed = XOR over p of (token[i-p] * layer_multiplier[p])     # uint64 wraparound
row_h = mixed % head_vocab_size[h] + head_offset[h]
```

`hash.py` reproduces this and is checked against a torch reimplementation of
`Qwen4ExpTextNGramEmbedding.forward` in `hash_selftest.py` — **0 mismatches in 32,000 entries**.

Two things bite anyone reimplementing it:
- The multiply needs **uint64 with wraparound**. `token × mult` reaches ~8e18, just under 2^63, so
  numpy int64 raises `OverflowError` where torch silently wraps.
- Context positions break at an EOS in the prefix, exactly as `_shift_right_ignore_eos` does.

## Finding 1: the four heads of an order share one address — it is a rank-4 factorization

All `head_vocab_sizes` are 4,880,000 and offsets are exact multiples of it, so `mixed % vocab` is
**identical** across heads 0–3 (and across 4–7). Verified on real text: "heads 0-3: IDENTICAL relative
row", "heads 4-7: IDENTICAL relative row".

So "4 heads per n-gram" is **not four independent hash functions**. It is one n-gram identity standing
in for a bigram/trigram, projected four ways through four learned 256-dim slabs — a rank-4 factorization
of a single 1024-dim signal. Consequences:

- You cannot decorrelate the heads by training table values; they are addressed together.
- Any "improve the hash" plan is really "retrain `key_proj`", because head count is `key_proj.weight`'s
  input width (`[8192, 2048]` → 2048/256 = 8 heads).
- Independent confirmation from `ple_head_values.py`: heads are **orthogonal** (cos ≈ 0.00, only 6.8% of
  rows byte-identical). Same address, different projections — exactly what a rank-4 factorization looks
  like, and the opposite of redundancy.

## Finding 2: head 3 and head 7 are sparse *by training*, not by coverage — but frequency matters

Zero masks: heads 0/1/2 are 100% identical to each other, 4/5/6 likewise. Head 3 agrees with head 0 on
only 26.4% of rows; head 7 with head 4 on 34.2%. Occupancy:

| head | kind | occupancy |
|---|---|---|
| 0–2 | bigram | 93.5% |
| 3 | bigram | **19.9%** |
| 4–6 | trigram | 100.0% |
| 7 | trigram | **34.2%** |

Coverage cannot explain a difference between heads that see identical addresses, so head 3's sparsity is
learned. But it is *partly* frequency-driven — liveness rises monotonically with bigram frequency:

| bigram count | h0–h2 | h3 | h4–h6 | h7 |
|---|---|---|---|---|
| 1 | 97% | 36% | 100% | 36% |
| 2–8 | 99% | 61% | 100% | 42% |
| 9–277 | 100% | 75% | 100% | 57% |

Even the most common bigrams in the corpus are only 75% live in head 3. And where head 3 *is* live it is
systematically quieter — RMS ≈ 0.4× its siblings (e.g. bigram 220→16: 0.173/0.181/0.145/**0.071**).
So head 3 was trained to be sparse *and* quiet: a deliberate down-weighting, not missing data.

## Finding 3: on real text, ~11% of the engram's output is dead

Per-token, over a chat corpus (20,000 tokens), share of each head's 256 dims reading an all-zero row:

| head | mean RMS | dead |
|---|---|---|
| 0 | 0.0260 | 0.5% |
| 1 | 0.0259 | 0.5% |
| 2 | 0.0238 | 0.5% |
| 3 | 0.0087 | 32.4% |
| 4 | 0.0077 | 0.0% |
| 5 | 0.0076 | 0.0% |
| 6 | 0.0066 | 0.0% |
| 7 | 0.0015 | 53.4% |

**223 of 2048 dims (10.9%) dead on average; 65.6% of tokens have at least one fully dead head; 34.4%
are fully live.** Note trigram heads 4–6 are *completely* live yet have lower RMS than the bigram heads —
the trigram signal is present but quieter, which is the opposite of what you would guess from occupancy
alone.

Live-row hit rate vs occupancy, both corpora (hit ≈ occupancy ⇒ the holes are unseen n-grams, benign):

| head | wiki hit / occ | chat hit / occ |
|---|---|---|
| 0–2 | 97.3% / 93.4% | 99.3% / 93.4% |
| 3 | 39.0% / 19.8% | 54.1% / 19.8% |
| 4–6 | 100.0% / 100.0% | 100.0% / 100.0% |
| 7 | 38.3% / 34.1% | 43.3% / 34.1% |

Working set is tiny: a 43.9k-token chat corpus touches **0.51%** of the bigram slabs and **0.72%** of the
trigram slabs — ~25k and ~35k distinct rows of 4.88M each. The table is enormous relative to the address
space any one prompt visits, which is why a hot-row cache is the right I/O fix.

## What can be improved, and what cannot

**Cannot, without a full retrain of `key_proj`:**
- head count (2×4) and the hash multipliers — they are weights, not config.
- making the four heads independent — they share an address by construction.

**Can, with table-value training only (cheap, no `key_proj` in the loop):**
- **Densify heads 3 and 7.** They are 20% / 34% live against siblings at 93% / 100% at the *same*
  addresses. This is the single largest defect: up to ~21% of the engram's dims are silent on a given
  token for no structural reason. Training objective: keep the layer-2 output predictive while forcing
  head 3/7 rows to carry their share of the signal.
- **Lift head 3's amplitude.** It runs at ~0.4× sibling RMS where live, so even a live head 3
  contributes a quarter of the energy its siblings do.
- **Balance the trigram heads' amplitude** relative to the bigram heads — 4–6 are fully populated but
  ~3× quieter than 0–2, which makes 8 equal-width dims unequal in effect.

**Storage, not quality:**
- IQ4_NL at the table level: 144 B/row, 5.62 GB, **8.05% median** relative error per row vs Q4_0's 9.40%
  at the same width. Nice because `PleTable::open()` is fully generic — a repacked table needs zero engine
  changes, and it makes the K-quant Whittle sizes loadable (their table type is a K-quant, which
  `kPleFormats[]` cannot decode).
- Zero-row skip: ~11% of gathers currently read a page of zeros.

## Tooling

| file | purpose |
|---|---|
| `hash.py` | the hash, uint64-exact |
| `hash_selftest.py` | proves `hash.py` == reference forward (0/32000 mismatches) |
| `hash_probe.py` | GGUF header reader (metadata + tensor infos + data offset) |
| `probe.py` | per-head live-row hit rate, occupancy, working set, zero-gather share |
| `head_coupling.py` | proves the within-order address identity and mask agreement |
| `energy.py` | per-token dead dims and RMS from the Q8_0 bytes |
| `freq_vs_live.py` | liveness vs n-gram frequency, to separate coverage from learned sparsity |

All read the table's **scale bytes only** via mmap — no dequant of 10 GB, no GPU. Each runs in well under
a minute on a CPU.
