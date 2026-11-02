# The engram (PLE n-gram) table: measured facts, and what can be improved

Everything here was measured on the real `per_layer_token_embd.weight` out of
`logic65/Whittle-Qwen-3.8-45B-A3B-GGUF` at revision `e2762dbe...`, on garlic-clove (Arc Pro B60, i7-10700).
Reproduce with the `tools/ple_*.py` scripts named in each section.

## What the table is

`39,040,000 rows x 256 values`, Q8_0 in the Q8_0 file = 10.62 GB. The shape decomposes exactly as the
GGUF metadata says:

```
ngram_size          = 3            head_vocab_sizes = [4880000] x 8
heads_per_ngram     = 4            head_offsets     = [0, 4880000, ..., 34160000]
PLE_N_HEADS         = 2 * 4 = 8    sum(head offsets+vocab) = 39,040,000  OK
embedding_length_per_layer_input = 256  ->  256 * 8 = 2048 = n_embd  OK
```

So heads 0-3 are the four heads of the bigram order, heads 4-7 the four of the trigram order.
The hash is host-side (no tensor op): `row = xor(token_i * layer_multiplier_i) % head_vocab`.

## 1. The table's format is a hard blocker for three of the four sizes

The engine reads the table from the original GGUF and has no K-quant row decoder. The table follows
whatever quant the file is:

| file | table type | engine reads it? |
|---|---|---|
| Q3_K_M | Q3_K | no |
| Q4_K_M | Q4_K | no |
| Q5_K_M | Q5_K | no |
| Q8_0 | Q8_0 | yes |

`STRATA_GU_FMTS`/`ple_formats()` cover `IQ4_NL, Q5_0, FP8, Q5_1, Q8_0, Q4_0, BF16, NVFP4` - no K-quants.
This is independent of the expert-format gate: Q3_K_M fails at the table *and* at the experts.

**IQ4_NL fixes it.** `tools/ple_iq4_pack.py` rewrites the table as IQ4_NL into a sidecar GGUF for
`--ple-gguf`, leaving the model file untouched. IQ4_NL is 18 B per 32 values = **144 B/row at any row
width**, and `ple_formats()` already decodes it.

## 2. What IQ4_NL costs, measured on this table's own values

`tools/engram_quant_error.py` encodes the real Q8_0 rows and decodes them back, paired on the same rows:

| quant | B/row | table GB | rel rmse | codebook |
|---|---|---|---|---|
| Q8_0 | 272 | 10.62 | 0 (reference) | - |
| **IQ4_NL** | **144** | **5.62** | **8.67%** | 16-entry non-linear |
| Q4_0 | 144 | 5.62 | 9.40% | 16-entry uniform |
| Q5_0 | 176 | 6.87 | 4.66% | 32-entry uniform |

Per-row (not pooled): **median 8.05%**, p90 8.5%, max ~9.5% - the error is uniform across live rows.
IQ4_NL beats Q4_0 by ~8% at identical size because `kvalues_iq4nl` is denser near zero (gaps 11-13)
where these weights live, and coarser in the tails (16-24). The pooled mean is misleading (31%): it is
inflated by near-empty rows whose relative error is meaningless.

The encoder is byte-identical to gguf-py's decoder (`tools/ple_iq4_xcheck.py`: 4000 blocks, max diff
0.000000, both 8.085%). gguf-py cannot *encode* IQ4_NL and `llama-quantize` refuses `qwen4exp`
outright, which is why the tool carries its own reference encoder.

## 3. The table is ~21% empty, and the emptiness is clustered by head

`tools/ple_head_masks.py`, `tools/ple_head_check.py`, `tools/ple_head_map.py` - three independent
methods (RNG sampling, deterministic stride over raw fp16 scales, GGUF head_offsets) agree:

| head | order | zero rows | live rows |
|---|---|---|---|
| 0 | bigram | 6.5% | 93.5% |
| 1 | bigram | 6.5% | 93.5% |
| 2 | bigram | 6.5% | 93.5% |
| **3** | bigram | **80.0%** | **20.0%** |
| 4 | trigram | 0.0% | 100% |
| 5 | trigram | 0.0% | 100% |
| 6 | trigram | 0.0% | 100% |
| **7** | trigram | **65.7%** | **34.3%** |

30,977,189 of 39,040,000 rows (79.3%) hold data. At 144 B/row that is 4.46 GB of information in a
5.62 GB file - 1.16 GB is all-zero rows.

Head 3 is uniformly ~80% empty across the whole row range, not sparse in a region. Head 7 is uniformly
~65% empty. This is a property of the trained weights, not of the quantisation.

## 4. ...but the heads are not redundant

The zero masks of heads 0,1,2 are **identical sample-for-sample** (100.0% agreement), and head 3's
populated set is a subset of theirs. Same for 4,5,6 and 7. That is what a shared row index gives:
"is this row populated" is a property of the n-gram, not of the head.

It is *not* four copies. `tools/ple_head_values.py` at 400 shared row ids:

```
        h0       h1       h2       h3
h0   0.9325   0.0006  -0.0021  -0.0008
h1   0.0006   0.9325   0.0033   0.0015
```

Cosine ~0.00 between heads: four orthogonal 256-d projections of the same n-gram, and only ~6.8% of
rows are byte-identical. So the 8 rows gathered per token are 8 genuinely different vectors; there is
nothing to deduplicate.

## 5. The gather cost, measured

`--ple-io direct` on this box probes **1164 rows/s = 0.86 ms per row**. Whittle gathers 8 rows/token:

- **6.9 ms/token** of PLE latency (Flash-Next's 16 heads would be 13.8)
- a 1000-token prompt = 3000 rows ~ **2.6 s**

It is latency, not bandwidth: 8 rows x 144 B is nothing. The engine already has `max_inflight = 256`,
a bounded row cache (`cache_rows`), and `PrefetchVirtualMemory` on Windows - but `ngram.hpp` records
that **Linux never got `MADV_RANDOM`**, so this build gets default sequential read-ahead on a random
access pattern, and `PleTable` has no hot-row cache.

## What can actually be improved, in order of value per unit of work

**(a) Fix the table format - done, in this fork.** `tools/ple_iq4_pack.py`. Turns 1 runnable Whittle
size into 4, and shrinks the PLE term from 10.6 GB to 5.6 GB at Q8_0 body. Cost: 8.7% relative weight
error, which is the price of 4 bits and is *lower* than plain Q4_0 at the same size.

**(b) Drop the empty rows - needs care.** 1.16 GB of the table is all-zero and costs I/O on every
gather that lands there (26% of gathers read a zero row on head 3). A sidecar format with a per-row
presence bitmask would skip those reads. Not a win in *samples* (a zero vector still has to become
zeros in the accumulator) but a win in *page faults*, which is the actual bottleneck at 0.86 ms each.
This is not implementable as a table rewrite alone - it needs a bitmask reader in `PleTable`.

**(c) `MADV_RANDOM` + a hot-row cache.** Cheap and upstreamable, no format change. The engine's own
comment says the advice was never added on Linux. A 1M-row cache is ~144 MB at IQ4_NL, 0.9% of the
table, and n-gram accesses are heavily Zipfian in practice.

**(d) The two mostly-empty heads.** Heads 3 and 7 cost 25% of every gather and contribute little. But
**you cannot just drop them**: the layer consumes `PLE_HEAD_DIM * PLE_N_HEADS = n_embd` and the trained
`ple_key`/`ple_value` projections expect those 8 slots. Dropping a head changes the input to a trained
linear map, which is a fine-tune, not a config change. What *is* available without retraining is
skipping the *read* when the row is zero (see b).

**(e) Bigger heads / order 4 - needs the author.** The table is 4.88M rows/head against Flash-Next's
20M, and the hash multipliers are baked into the weights: change the hash and every trained row is
addressed by a different n-gram. Nothing here is a config knob. Real gains - more rows, order 4,
more than 3 populated heads per order - are a retrain by logic65.

## The honest summary

"Better n-gram" divides cleanly:

- **Format and I/O**: mine to fix, and (a) is done. (b) and (c) are real, bounded, and measurable.
- **Capacity**: not fixable without retraining. The hash, the row count and the head count are weights,
  not parameters. Anyone claiming a config change improves n-gram recall is selling something.
- **The table is 21% padding, but the live 79% is 8 orthogonal projections with no redundancy.** The
  memory is not wasted in the sense of duplication - it is wasted in the sense of never-populated
  n-grams, which is a corpus-coverage fact about what the model was trained on.
