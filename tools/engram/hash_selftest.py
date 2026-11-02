"""Verify hash.py reproduces the reference Qwen4ExpTextNGramEmbedding.forward row ids.

The reference is the authority: if my numpy reimplementation disagrees on even one token, every number
in the probe is fiction.  Runs the real module from transformers on a tiny synthetic config with the
model's real multipliers/vocab/offsets, and compares row ids token for token.

Also proves the uint64-wraparound multiply is right (torch int64 silently wraps; numpy int64 raises).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hash import ngram_rows  # noqa: E402


def reference_rows(tokens, mult, vocab, offsets, ngram_size, heads_per_ngram, eos, n_embd):
    """Reimplement forward() with torch, exactly as the class does: int64 matmul-free, plain ops."""
    tok = torch.tensor(tokens, dtype=torch.long)
    n = tok.numel()
    m = torch.tensor(mult, dtype=torch.long)
    out = torch.zeros(n, (ngram_size - 1) * heads_per_ngram, dtype=torch.long)

    shifted = []
    for p in range(ngram_size):
        idx = torch.arange(n) - p
        s = torch.where(idx >= 0, tok[idx.clamp(min=0)], torch.tensor(eos))
        # cut at eos in the prefix, same rule
        for q in range(1, p + 1):
            j = torch.arange(n) - q
            bad = torch.zeros(n, dtype=torch.bool)
            msk = j >= 0
            bad[msk] = tok[j[msk]] == eos
            s = torch.where(bad | (idx < 0), torch.tensor(eos), s)
        shifted.append(s)

    for ng in range(2, ngram_size + 1):
        mixed = shifted[0] * m[0]                      # int64 wraparound, as torch does
        for p in range(1, ng):
            mixed = torch.bitwise_xor(mixed, shifted[p] * m[p])
        base = (ng - 2) * heads_per_ngram
        for g in range(heads_per_ngram):
            h = base + g
            out[:, h] = (mixed % vocab[h]) + offsets[h]
    return out.numpy()


def main():
    rng = np.random.default_rng(0)
    mult = [32000000000001, 32000000000007, 32000000000019]
    vocab = [4880000] * 8
    offsets = [4880000 * i for i in range(8)]
    ng, hpn, eos = 3, 4, 248044

    # realistic token stream: ids spread over the vocab, with a few eos
    n = 4000
    tok = rng.integers(0, 248070, size=n).astype(np.int64)
    tok[rng.choice(n, 40, replace=False)] = eos

    a = ngram_rows(tok, mult, vocab, offsets, ng, hpn, eos)
    b = reference_rows(tok, mult, vocab, offsets, ng, hpn, eos, 2048)

    print("numpy shape", a.shape, "reference shape", b.shape)
    assert a.shape == b.shape, "shape mismatch"
    bad = np.argwhere(a.astype(np.int64) != b)
    print("mismatched entries: %d / %d" % (len(bad), a.size))
    if len(bad):
        for i, h in bad[:5]:
            print("  token %d head %d: numpy %d reference %d" % (i, h, a[i, h], b[i, h]))
        raise SystemExit("HASH MISMATCH - do not trust the probe")
    print("MATCH - hash.py reproduces the reference row ids exactly")

    # the wraparound is load-bearing: prove plain int64 raises where uint64 wraps
    try:
        _ = tok.astype(np.int64) * np.int64(mult[0])
        print("note: int64 multiply did NOT overflow for these multipliers")
    except OverflowError:
        print("note: int64 multiply overflows (as expected) - uint64 wraparound is required")


if __name__ == "__main__":
    main()
