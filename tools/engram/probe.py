"""Per-head live-row hit rate and working set on a real token stream (mmap; no per-row seeks).

    python probe.py --gguf <w45_Q8_0.gguf> --tokens data/tokens_wiki.npy

For each head:
  hit%   of the DISTINCT rows this head actually addresses on this corpus, the share that are live
  occ%   that head's overall live-row share (sampled from the whole head)
  touch% distinct rows addressed / rows in the head - the ceiling on what a finetune could ever reach

hit ~= occ  -> the holes are n-grams the model never addresses; benign, the model routes around them.
hit << occ  -> the hash steers real text into empty rows; training table values cannot fix that,
               because a row that is never selected never receives a gradient.
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
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=53)
    a = ap.parse_args()

    meta, infos, ds = read_gguf(a.gguf)
    mult = meta["qwen4exp.ple.layer_multipliers"]
    vocab = meta["qwen4exp.ple.head_vocab_sizes"]
    offsets = meta["qwen4exp.ple.head_offsets"]
    ng = meta.get("qwen4exp.ple.ngram_size", 3)
    hpn = meta.get("qwen4exp.ple.heads_per_ngram", 4)
    eos = meta.get("qwen4exp.ple.eos_token_id", 248044)
    dims, tt, off = infos["per_layer_token_embd.weight"]
    head_dim, n_rows = int(dims[0]), int(dims[1])
    nblk = head_dim // 32
    rb = nblk * 34

    tok = np.load(a.tokens).astype(np.int64).ravel()
    if a.limit and len(tok) > a.limit:
        tok = tok[: a.limit]

    print("table   : %d rows, head_dim %d, %d B/row" % (n_rows, head_dim, rb))
    print("metadata: ngram %d, heads_per_ngram %d, eos %d" % (ng, hpn, eos))
    print("tokens  : %d (distinct %d) from %s\n" % (len(tok), len(np.unique(tok)), Path(a.tokens).name))

    rows = ngram_rows(tok, mult, vocab, offsets, ng, hpn, eos)

    # one mmap of the whole table: scale bytes are at block starts, 34 B apart
    mm = np.memmap(a.gguf, dtype=np.uint8, mode="r")
    base_ptr = ds + off
    nhead = len(vocab)

    def live(rids):
        """bool mask: is the fp16 scale of the first block of each row non-zero?"""
        byte = base_ptr + rids.astype(np.int64) * rb
        return (mm[byte].astype(np.uint16) | mm[byte + 1].astype(np.uint16)) != 0

    print("  %-5s %-8s %8s %8s %9s %12s" % ("head", "kind", "hit%", "occ%", "touch%", "distinct"))
    hits = []
    for h in range(nhead):
        col = rows[:, h].astype(np.int64)
        uniq = np.unique(col)
        hl = live(uniq)
        hit = 100.0 * float(hl.mean())
        # occupancy of the whole head, strided
        sample = np.arange(offsets[h], offsets[h] + vocab[h], a.stride, dtype=np.int64)
        occ = 100.0 * float(live(sample).mean())
        touch = 100.0 * len(uniq) / vocab[h]
        kind = "bigram" if h < hpn else "trigram"
        print("  %-5d %-8s %7.1f%% %7.1f%% %8.3f%% %12d" % (h, kind, hit, occ, touch, len(uniq)))
        hits.append(hit)

    print("\nmean live-row hit rate: %.1f%%" % (np.mean(hits)))

    z = np.zeros(rows.shape[0], dtype=np.int32)
    for h in range(nhead):
        z += (~live(rows[:, h].astype(np.int64))).astype(np.int32)
    print("gathers reading >=1 zero row: %.1f%%" % (100 * float((z > 0).mean())))
    print("mean zero rows per gather   : %.2f of %d" % (float(z.mean()), nhead))
    print("tokens where ALL 8 rows zero: %.1f%%" % (100 * float((z == nhead).mean())))
    del mm


if __name__ == "__main__":
    main()
