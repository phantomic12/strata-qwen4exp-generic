"""Is head 3 / head 7 sparsity a coverage artifact, or a deliberate property of the weights?

All four heads of an order address the SAME relative bucket, so any bigram the corpus contains lands in
exactly the same bucket for heads 0-3.  Yet heads 0-2 are 93.5% live and head 3 is 19.9% live.  Coverage
cannot explain a difference between heads that see identical addresses.

Two competing stories:
  (a) COVERAGE  - head 3 is empty where the training corpus had no n-gram.  Then head 3's zero set would
                  equal heads 0-2's zero set (it is a strict superset, so: no) AND rare n-grams would be
                  zero more often than common ones.
  (b) LEARNED   - head 3 is deliberately sparse/quiet at addresses its siblings fill.  Then liveness is
                  uncorrelated with n-gram frequency.

Test: bucket the observed bigrams by frequency, and report head liveness and RMS per bucket.  If rare
bigrams are dead and common ones live in EVERY head, it is coverage.  If head 3 is dead even on the most
common bigrams, it is learned.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hash_probe import read_gguf  # noqa: E402
from hash import ngram_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokens", required=True)
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
    nblk = head_dim // 32
    rb = nblk * 34

    tok = np.load(a.tokens).astype(np.int64).ravel()
    rows = ngram_rows(tok, mult, vocab, offsets, ng, hpn, eos).astype(np.int64)

    # bigram frequency over the token stream (skip anything spanning an eos)
    pairs = Counter()
    for i in range(1, len(tok)):
        if tok[i] != eos and tok[i - 1] != eos:
            pairs[(int(tok[i - 1]), int(tok[i]))] += 1
    print("distinct bigrams in corpus: %d over %d tokens" % (len(pairs), len(tok)))

    mm = np.memmap(a.gguf, dtype=np.uint8, mode="r")

    def stats(rids):
        """(live bool, rms float) per row."""
        base = ds + off + rids * rb
        idx = base[:, None] + np.arange(rb)[None, :]
        b = mm[idx].reshape(len(rids), nblk, 34)
        d = b[:, :, 0:2].copy().view(np.float16).reshape(len(rids), nblk).astype(np.float32)
        q = b[:, :, 2:34].copy().view(np.int8).astype(np.float32)
        v = (d[:, :, None] * q).reshape(len(rids), -1)
        r = np.sqrt((v ** 2).mean(axis=1))
        return r > 0, r

    # bucket every (token index) that has a complete bigram, by that bigram's frequency
    idxs = [i for i in range(1, len(tok)) if tok[i] != eos and tok[i - 1] != eos]
    freq = np.array([pairs[(int(tok[i - 1]), int(tok[i]))] for i in idxs])
    sel = np.array(idxs)
    qs = np.percentile(freq, [0, 20, 40, 60, 80, 100])
    print("bigram frequency percentiles: %s\n" % np.round(qs, 1).tolist())

    print("  %-22s %6s | %s" % ("freq bucket", "n", "  ".join("h%d" % h for h in range(8))))
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (freq >= lo) & (freq <= hi)
        if not m.any():
            continue
        sub = sel[m]
        cells = []
        for h in range(8):
            live, rms = stats(rows[sub, h])
            cells.append("%.0f%%" % (100 * live.mean()))
        print("  %-22s %6d | %s" % ("%.1f-%.1f" % (lo, hi), m.sum(), "  ".join(cells)))

    # and the very top: most common 20 bigrams, are them live in head 3?
    top = [p for p, _ in pairs.most_common(20)]
    print("\nmost common bigrams, liveness in heads 0-3 (same address for all):")
    for a_, b_ in top[:10]:
        i = next((j for j in range(1, len(tok)) if int(tok[j-1]) == a_ and int(tok[j]) == b_), None)
        if i is None:
            continue
        cell = []
        for h in range(4):
            live, rms = stats(rows[[i], h])
            cell.append("%s(%.4f)" % ("L" if live[0] else "Z", rms[0]))
        print("   %6d->%-6d n=%-4d %s" % (a_, b_, pairs[(a_, b_)], "  ".join(cell)))
    del mm


if __name__ == "__main__":
    main()
