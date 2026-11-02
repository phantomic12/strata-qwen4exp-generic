#!/usr/bin/env python3
"""Do the four heads of one n-gram order store the same values, or four different projections?

The zero masks of heads 0,1,2 are identical sample-for-sample, and head 3's populated set is a SUBSET of
theirs.  Same for 4,5,6 and 7.  That is what you get if all heads of an order share a row index (same
multiplier, per-head block offset), so "is this row populated" is a property of the n-gram rather than of
the head.

The follow-on question decides whether any of this is shrinkable: at a shared row index, do the heads hold
the SAME 256 values (redundant copies) or DIFFERENT ones (four projections of one n-gram)?  Compared with
cosine similarity and an exact-equality count.
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
from gguf.quants import dequantize  # noqa: E402

from ple_iq4_pack import find_table  # noqa: E402

HEADS = 8


def main():
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.home() / "models/w45_Q8_0.gguf")
    ds, (dims, tt, off) = find_table(path)
    head_dim, total = int(dims[0]), int(dims[1])
    bs, ts = GGML_QUANT_SIZES[QT(tt)]
    rb = (head_dim // bs) * ts
    rph = total // HEADS
    if QT(tt).name != "Q8_0":
        sys.exit("needs Q8_0")

    def row(h, r):
        with open(path, "rb") as f:
            f.seek(ds + off + (h * rph + r) * rb)
            raw = f.read(rb)
        return np.asarray(dequantize(np.frombuffer(raw, dtype=np.uint8), QT(tt)),
                          dtype=np.float32).reshape(head_dim)

    rng = np.random.default_rng(17)
    n = 400
    ids = np.sort(rng.integers(0, rph, n))
    print("%s: %d rows/head, %d shared row ids\n" % (path.name, rph, n))

    for group, heads in (("bigram order", [0, 1, 2, 3]), ("trigram order", [4, 5, 6, 7])):
        got = {h: np.stack([row(h, r) for r in ids]) for h in heads}
        print(group)
        print("  pairwise, over the same %d row ids:" % n)
        print("        " + "".join("%10s" % ("h%d" % j) for j in heads))
        for i in heads:
            cells = []
            for j in heads:
                a, b = got[i], got[j]
                cos = (a * b).sum(axis=1) / np.maximum(
                    np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)
                cells.append("%9.4f" % float(np.mean(cos)))
            print("    h%-3d%s" % (i, "".join(cells)))
        print("  exact byte equality:")
        for j in heads[1:]:
            eq = float((got[heads[0]] == got[j]).all(axis=1).mean())
            print("    h%d == h%d : %.1f%% of rows identical" % (heads[0], j, 100 * eq))
        print()


if __name__ == "__main__":
    main()
