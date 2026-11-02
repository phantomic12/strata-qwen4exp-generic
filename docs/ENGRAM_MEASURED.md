# The engram, measured on a verified load

Supersedes the per-head numbers in `ENGRAM_ANATOMY.md` and the recommendations in `ENGRAM_COST.md`.
Those numbers were right; the first *forward-pass* measurement of them was wrong, and this file records
both the bug and the corrected result.

## The loader bug: `from_pretrained` silently discards the engram table

`Qwen4ExpTextNGramEmbedding.__init__` derives the bucket layout from config as **distinct primes**:

```python
size = _find_nth_prime_after(ngram_vocab_size_base - 1, global_head_idx + 1)
# [4880011, 4880021, 4880059, 4880069, 4880087, 4880089, 4880101, 4880111]
# sum 39,040,548  ->  padded to 39,040,640
```

The trained checkpoint stores **equal sizes**:

```
model.ple_embedding.ngram_heads_vocab_sizes  I64 [8]  = [4880000] * 8   sum 39,040,000
model.ngram_embedding.shard_{0..4}.weight    total rows 39,040,000
GGUF per_layer_token_embd.weight             [256, 39040000]
```

39,040,640 ≠ 39,040,000, so the embedding is built at the wrong size, the checkpoint tensor fails to
assign, transformers logs `UNEXPECTED` and **leaves the embedding at its random initialisation**. The
model then runs perfectly happily — gate, projections, conv, loss — against random rows.

Measured (`engram_layout.py`):

```
raw table     (39040000, 256)   occupancy [93.4, 93.4, 93.4, 20.5, 99.9, 99.9, 99.9, 34.7]
module table  (39040640, 256)   rows identical? False    live 100.0%
```

Why this matters beyond one bad run:

- Any engram fine-tune started the obvious way **trains noise**.
- The published `transformers` reference does not reproduce these weights at all.
- It produced a **false negative** for me: an 8-head "all identical at 0.32, nothing empty, gate 0.503"
  result, which read as a clean scientific finding and was pure artefact of the random init.

**Fix (`tools/engram/engram_load.py`).** Read the shards directly, install the trained layout buffers,
place the rows, then *verify*: a trained table has runs of all-zero rows, a fresh `nn.Embedding` has
none, so reproducing `[92.4, 92.4, 92.4, 20.2, 100.0, 100.0, 100.0, 32.9]` per-head occupancy cannot
happen by accident. The loader raises rather than continue if the check fails.

## Corrected per-head measurement (corrected head count)

Note `ngram_heads` is 8 (4 bigram + 4 trigram), not the 16 previously written in one place.

1024 tokens of chat text, table frozen, verified load:

| head | kind | gather norm | empty | **share of engram energy** |
|---|---|---|---|---|
| 0 | bigram | 0.4015 | 0.6% | **30.7%** |
| 1 | bigram | 0.4014 | 0.6% | **30.4%** |
| 2 | bigram | 0.3693 | 0.6% | **26.9%** |
| 3 | bigram | 0.1360 | 29.3% | **5.3%** |
| 4 | trigram | 0.1256 | 0.0% | **2.2%** |
| 5 | trigram | 0.1246 | 0.0% | **2.2%** |
| 6 | trigram | 0.1076 | 0.0% | **1.9%** |
| 7 | trigram | 0.0268 | 49.5% | **0.3%** |

**Heads 0–2 carry 88.0% of the signal. Heads 4–7 carry 6.6% combined; head 7 carries 0.3%** while being
49.5% empty. The layer-2 input is 2048 wide and roughly 92% of that width is doing almost nothing.

### The gate is open — it is not the obstruction

```
gate mean 0.711, std 0.302, min 0.0096, max 0.976
> 0.5 on 79.9% of positions,  < 0.1 on 10.2%
per hc-copy: [0.673, 0.668, 0.722, 0.781]
```

So the quiet heads are quiet because **their rows are empty or low-norm**, not because the model
suppresses them. This kills the targeted-weight-edit plan: there is nothing to unmute. It also means the
"signed-sqrt gate" borrow from V4.1 is moot — Whittle already has that exact gate, and it is working.

Baseline loss on this sample: **1.2525**. Any retrain must beat that to mean anything.

## What the retrain is actually for

Not coverage, not a gate. The engram's 2048 dimensions are allocated uniformly but used very unevenly.
Levers in order:

1. **Head 7 and head 3 are half-dead** (49.5% / 29.3% empty) and contribute 0.3% / 5.3%. Either fill them
   (stage 2 coverage) or stop reserving 512 of 2048 dims for them. Reassigning dimensions is a
   `key_proj` change and therefore a retrain, but a much smaller one than growing the table.
2. **Trigram amplitude is ~3× below bigram** — heads 4–6 are fully populated yet quiet. That is a
   training-signal artefact, fixable by table-value training without touching the hash.
3. **Capacity.** Whittle has 2.16 live table values per body parameter against V4.1's ~8.3 per layer. No
   finetune fixes a capacity ratio; that is the case for a larger table.

## Cost (unchanged, see ENGRAM_COST.md for the derivation)

- 300M tokens: **$17–31 on H100**, $18–32 on L40S, $34–61 on A100-80.
- Stages 1–2 (table values, then `key_proj`/`value_proj`) fit inside Modal's **$30/month free compute**.
- Data, not FLOPs, sets the bill: measured `distinct bigrams ~ n^0.819`, so 90% fill of a 4.88M-bucket
  head needs ~71M unique tokens.

## Files

| file | purpose |
|---|---|
| `engram_load.py` | **the loader that works** — places the real table and verifies it |
| `engram_layout.py` | the proof that `from_pretrained` discards the table |
| `engram_hashcheck.py` | proves `hash.py` matches the model's own row ids exactly |
| `engram_gate2.py` | the corrected gate/energy measurement above |
| `hash.py` | the hash, uint64-exact, verified 0/32000 mismatches vs the reference forward |
| `probe.py`, `energy.py`, `head_coupling.py`, `freq_vs_live.py` | offline table statistics (all still valid) |
| `coverage_curve.py`, `compress_probe.py` | data-volume and compressed-vocab measurements |
