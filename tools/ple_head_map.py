#!/usr/bin/env python3
"""Heads 3 and 7 are ~81% and ~65% empty; the others are ~7% or 0%.  Is that the 4th head of each order?

PLE_N_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM = 2 orders x 4 heads.  So heads 0-3 are the four heads
of the bigram order and 4-7 the four of the trigram order.  The empty heads are index 3 and index 7 -
i.e. index 3 WITHIN each group.  That would say the model effectively runs on 3 of its 4 declared heads
per n-gram order, which is a shape fact worth knowing before anyone sizes a table or reasons about what
the "10 B memory" holds.

Tests:
  * is the emptiness a contiguous tail range of rows within the head, or scattered?
  * do the empty rows start at a round number (a trained/unpopulated boundary)?
  * is it identical for the other Whittle sizes (i.e. baked into the weights, not the quantisation)?
"""
import struct
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths  # noqa: E402

_paths.add_gguf_py()
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402

from ple_iq4_pack import find_table  # noqa: E402


def zero_at(f, ds, off, rb, ts, nblk, row):
    f.seek(ds + off + row * rb)
    raw = f.read(rb)
    return all(struct.unpack("<e", raw[k * ts:k * ts + 2])[0] == 0.0 for k in range(nblk))


def main():
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.home() / "models/w45_Q8_0.gguf")
    ds, (dims, tt, off) = find_table(path)
    head_dim, total = int(dims[0]), int(dims[1])
    bs, ts = GGML_QUANT_SIZES[QT(tt)]
    rb = (head_dim // bs) * ts
    nblk = head_dim // bs
    if QT(tt).name != "Q8_0":
        sys.exit("byte check expects Q8_0; %s rows have no fp16 scale to inspect" % QT(tt).name)
    nheads = 8
    rph = total // nheads
    print("%s: %d rows, %d B/row, %d rows/head\n" % (path.name, total, rb, rph))

    with open(path, "rb") as f:
        for h in range(nheads):
            # a coarse map: 100 buckets across the head, zero-rate in each
            buckets = []
            for b in range(100):
                lo = h * rph + b * (rph // 100)
                # 40 rows spread over the bucket
                step = max(1, (rph // 100) // 40)
                z = sum(zero_at(f, ds, off, rb, ts, nblk, lo + i * step) for i in range(40))
                buckets.append(z / 40)
            rate = 100 * sum(buckets) / len(buckets)
            # where does it change?  first bucket and last bucket, plus a profile
            prof = "".join("#" if v > 0.9 else "+" if v > 0.1 else "." for v in buckets)
            print("  head %d  %5.1f%% zero   %s" % (h, rate, prof))
    print("\n  legend: '.' <10% zero, '+' 10-90%, '#' >90%; left to right is low row id to high")


if __name__ == "__main__":
    main()
