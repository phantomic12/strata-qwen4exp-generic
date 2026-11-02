"""How much engram signal is actually lost per token on real text?

The tables are Q8_0: per 32 values an fp16 scale d and 32 int8 quants, value = d * q.

For each token we gather 8 rows (4 bigram slabs + 4 trigram slabs).  A row whose scales are all zero
contributes exactly nothing.  So the honest question is not "how many rows are zero" but "how much of
the 2048-dim engram vector is dead, on real text".

Reports, per head: mean RMS of the gathered row, and per token the share of the head's 256 dims that
are dead.  Then the same across all 2048 dims - the fraction of the engram output that is zeros.
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
    ap.add_argument("--limit", type=int, default=20000)
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
    if a.limit and len(tok) > a.limit:
        tok = tok[: a.limit]
    rows = ngram_rows(tok, mult, vocab, offsets, ng, hpn, eos).astype(np.int64)

    mm = np.memmap(a.gguf, dtype=np.uint8, mode="r")
    nhead = len(vocab)

    def rms(rids):
        """per-row RMS of the dequantized 256-dim vector, vectorised over rows."""
        base = ds + off + rids * rb
        # (n_rows, nblk, 34) view of the row bytes
        idx = base[:, None] + np.arange(rb)[None, :]
        b = mm[idx].reshape(len(rids), nblk, 34)
        d = b[:, :, 0:2].copy().view(np.float16).reshape(len(rids), nblk).astype(np.float32)
        q = b[:, :, 2:34].copy().view(np.int8).astype(np.float32)
        v = d[:, :, None] * q
        return np.sqrt((v.reshape(len(rids), -1) ** 2).mean(axis=1))

    print("tokens %d, table %d rows, %d B/row\n" % (len(tok), int(dims[1]), rb))
    print("  %-5s %-8s %10s %10s %10s" % ("head", "kind", "mean RMS", "median", "dead%"))
    dead = np.zeros(len(tok))
    for h in range(nhead):
        r = rms(rows[:, h])
        d = 100.0 * float((r == 0).mean())
        dead += (r == 0)
        kind = "bigram" if h < hpn else "trigram"
        print("  %-5d %-8s %10.5f %10.5f %9.1f%%"
              % (h, kind, r.mean(), np.median(r), d))
    del mm

    print("\ndead dims per token: %.1f of 2048 (%.1f%%)"
          % (dead.mean() * head_dim, 100 * dead.mean() / nhead))
    print("tokens with any dead head : %.1f%%" % (100 * float((dead > 0).mean())))
    print("tokens with all heads live: %.1f%%" % (100 * float((dead == 0).mean())))


if __name__ == "__main__":
    main()
