#!/usr/bin/env python3
"""Do the live heads share which rows are populated?  And how much of the table is actually information?

Heads 0, 1 and 2 showed the same zero/not-zero pattern bucket for bucket, head 3 was uniformly ~80% empty
and head 7 ~65% empty while 4-6 were full.  If two heads are zero on exactly the same row ids, that
says the zero-ness is a property of the n-gram (the row index) and not of the head, which is what you
would expect if an n-gram simply never appeared in training - and it means the empty rows are shared, so
the "wasted" space is counted once, not once per head.

Measures, over an evenly spaced sample of the full 4,880,000 rows per head:
  * the zero mask per head, and the pairwise agreement between heads
  * rows where ALL heads are zero (a genuinely unused n-gram) vs SOME heads
  * the table's non-zero row count and what it costs at 144 B/row
"""
import struct
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths  # noqa: E402

_paths.add_gguf_py()
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402

from ple_iq4_pack import find_table  # noqa: E402

HEADS = 8


def main():
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.home() / "models/w45_Q8_0.gguf")
    ds, (dims, tt, off) = find_table(path)
    head_dim, total = int(dims[0]), int(dims[1])
    bs, ts = GGML_QUANT_SIZES[QT(tt)]
    rb = (head_dim // bs) * ts
    nblk = head_dim // bs
    if QT(tt).name != "Q8_0":
        sys.exit("needs Q8_0 (fp16 scales)")
    rph = total // HEADS
    stride = 233
    n = rph // stride
    print("%s: %d rows/head, sampling every %dth row -> %d samples/head\n" % (path.name, rph, stride, n))

    masks = np.zeros((HEADS, n), dtype=bool)          # True = zero row
    with open(path, "rb") as f:
        for h in range(HEADS):
            base = h * rph
            for i in range(n):
                f.seek(ds + off + (base + i * stride) * rb)
                raw = f.read(rb)
                masks[h, i] = all(struct.unpack("<e", raw[k * ts:k * ts + 2])[0] == 0.0
                                  for k in range(nblk))

    print("zero rate per head: " + "  ".join("h%d %.1f%%" % (h, 100 * masks[h].mean()) for h in range(HEADS)))
    print()
    print("pairwise agreement of the zero mask (share of samples where two heads agree):")
    print("      " + "".join("%7s" % ("h%d" % j) for j in range(HEADS)))
    for i in range(HEADS):
        row = "".join("%6.1f%%" % (100 * (masks[i] == masks[j]).mean()) for j in range(HEADS))
        print("  h%d  %s" % (i, row))
    print()
    nz = ~masks
    all_zero = masks.all(axis=0)
    any_zero = masks.any(axis=0)
    print("rows where EVERY head is zero : %6.2f%% of sampled n-grams" % (100 * all_zero.mean()))
    print("rows where SOME head is zero  : %6.2f%%" % (100 * any_zero.mean()))
    print("rows where every head has data: %6.2f%%" % (100 * nz.all(axis=0).mean()))
    print()
    # non-zero rows per head, as a whole-table count
    for h in range(HEADS):
        live = int(round((1 - masks[h].mean()) * rph))
        print("  head %d: %9d non-zero rows of %d  (%.1f%%)  -> %.2f GB at 144 B/row"
              % (h, live, rph, 100 * (1 - masks[h].mean()), live * 144 / 1e9))
    total_live = sum(int(round((1 - masks[h].mean()) * rph)) for h in range(HEADS))
    print("\n  all heads: %d non-zero rows of %d (%.1f%%), %.2f GB at 144 B/row"
          % (total_live, total, 100 * total_live / total, total_live * 144 / 1e9))


if __name__ == "__main__":
    main()
