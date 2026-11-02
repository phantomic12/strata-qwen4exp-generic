"""Are the four heads within an n-gram order independent, or do they all address the SAME relative row?

Every head in an order computes the same `mixed` value, then row = mixed % vocab[h] + offset[h].
With all head_vocab_sizes equal (4,880,000) and offsets spaced by exactly that, the relative row
(mixed % vocab) is IDENTICAL across the four heads - the heads differ only in which 256-wide slab of
the table they read.  The probe's identical distinct-row counts for heads 0/1/2 and 4/5/6 are the
signature of exactly that.

If true, then "4 heads" is not four independent hash functions.  It is one hash identity standing in
for an n-gram, with four different learned 256-dim projections of it - a rank-4 factorisation.  That
changes what retraining can do: you cannot decorrelate the heads by training table values, because
they are addressed together.

Also re-checks the zero-mask claim from ple_head_masks.py (identical masks within an order), which
contradicts the measured occupancies (93.4 / 93.4 / 93.4 / 19.8).  One of the two is wrong and the
diagnosis depends on which.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hash_probe import read_gguf  # noqa: E402
from hash import ngram_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--n", type=int, default=20000, help="rows to compare masks on")
    a = ap.parse_args()

    meta, infos, ds = read_gguf(a.gguf)
    mult = meta["qwen4exp.ple.layer_multipliers"]
    vocab = meta["qwen4exp.ple.head_vocab_sizes"]
    offsets = meta["qwen4exp.ple.head_offsets"]
    ng = meta.get("qwen4exp.ple.ngram_size", 3)
    hpn = meta.get("qwen4exp.ple.heads_per_ngram", 4)
    eos = meta.get("qwen4exp.ple.eos_token_id", 248044)
    dims, tt, off = infos["per_layer_token_embd.weight"]
    head_dim = int(dims[0])
    rb = (head_dim // 32) * 34

    print("head_vocab_sizes:", vocab)
    print("head_offsets    :", offsets)

    # 1. relative row identity, straight from the hash on real text
    tok = np.load(a.tokens).astype(np.int64).ravel()
    rows = ngram_rows(tok, mult, vocab, offsets, ng, hpn, eos).astype(np.int64)
    rel = rows - np.array(offsets, dtype=np.int64)
    print("\nrelative-row identity across heads of one order:")
    for base in range(0, len(vocab), hpn):
        grp = rel[:, base:base + hpn]
        same = all(np.array_equal(grp[:, 0], grp[:, g]) for g in range(1, hpn))
        print("  heads %d-%d: %s" % (base, base + hpn - 1,
                                     "IDENTICAL relative row" if same else "independent"))
        if not same:
            print("    e.g. token 0:", grp[0].tolist())

    # 2. zero masks: are the same relative rows empty in every slab of an order?
    mm = np.memmap(a.gguf, dtype=np.uint8, mode="r")
    base_ptr = ds + off

    def live(rids):
        b = base_ptr + rids.astype(np.int64) * rb
        return (mm[b].astype(np.uint16) | mm[b + 1].astype(np.uint16)) != 0

    sample_rel = np.arange(0, min(vocab[0], 4880000), max(1, 4880000 // a.n))
    print("\nzero-mask agreement over %d relative rows:" % len(sample_rel))
    for base in range(0, len(vocab), hpn):
        masks = [live(sample_rel + offsets[base + g]) for g in range(hpn)]
        occup = [100.0 * float(m.mean()) for m in masks]
        print("  heads %d-%d occupancy %s" % (base, base + hpn - 1,
                                              " ".join("%.1f%%" % o for o in occup)))
        for g in range(1, hpn):
            agree = 100.0 * float((masks[0] == masks[g]).mean())
            print("    head %d vs %d: masks identical on %.1f%% of rows" % (base, base + g, agree))
    del mm


if __name__ == "__main__":
    main()
