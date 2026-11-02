"""The hash, exactly as Qwen4ExpTextNGramEmbedding.forward computes it.

    mixed = xor over positions p in 0..ngram-1 of (shifted_token[p] * layer_multiplier[p])
    row   = mixed % head_vocab_size[head] + head_offset[head]

Heads are laid out as `ngram-2` blocks of `heads_per_ngram`: 0..3 hash the bigram, 4..7 the trigram.

The multiply MUST be uint64 with wraparound: the reference runs it on torch int64 tensors, where the
product silently wraps. The multipliers are up to ~3.2e13 and token ids up to 2.5e5, so token*mult
reaches 8e18 - just under 2^63 - and a plain numpy int64 multiply raises OverflowError rather than
wrapping. uint64 reproduces torch's behaviour bit for bit (verified against the reference in
hash_selftest.py).
"""
import numpy as np

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def shifted_tokens(tok, ngram_size, eos):
    """shifted[p][i] = token (i-p), or eos where the context is broken - matching _shift_right_ignore_eos.

    The reference cuts at an eos in the PREFIX: for shift p, position i is valid only if i-p >= 0 and no
    eos appears in (i-p, i].  Where invalid, the value is eos (and that eos then propagates into longer
    shifts, because the longer shift tests a superset of positions).
    """
    n = len(tok)
    out = []
    for p in range(ngram_size):
        if p == 0:
            out.append(tok.astype(np.int64).copy())
            continue
        idx = np.arange(n) - p
        valid = idx >= 0
        src = np.where(valid, tok[np.clip(idx, 0, n - 1)], eos)
        # any eos in (i-p, i] breaks the context for shift p
        for q in range(1, p + 1):
            j = np.arange(n) - q
            m = j >= 0
            bad = np.zeros(n, dtype=bool)
            bad[m] = tok[j[m]] == eos
            src = np.where(bad | ~valid, eos, src)
            valid = valid & ~bad
        out.append(src.astype(np.int64))
    return out


def ngram_rows(tokens, mult, vocab, offsets, ngram_size=3, heads_per_ngram=4, eos=248044):
    """(n_tokens, n_heads) uint32 row ids, matching the reference forward pass."""
    tok = np.asarray(tokens, dtype=np.int64).ravel()
    n = len(tok)
    nheads = (ngram_size - 1) * heads_per_ngram
    sh = shifted_tokens(tok, ngram_size, eos)
    out = np.zeros((n, nheads), dtype=np.uint32)

    for ngram in range(2, ngram_size + 1):
        mixed = (sh[0].astype(np.uint64) * np.uint64(mult[0])) & MASK64
        for p in range(1, ngram):
            mixed = mixed ^ ((sh[p].astype(np.uint64) * np.uint64(mult[p])) & MASK64)
        base = (ngram - 2) * heads_per_ngram
        for g in range(heads_per_ngram):
            h = base + g
            out[:, h] = ((mixed % np.uint64(vocab[h])) + np.uint64(offsets[h])).astype(np.uint32)
    return out
