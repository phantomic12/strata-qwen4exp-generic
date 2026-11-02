#!/usr/bin/env python3
"""Which heads of the engram table hold near-empty rows, and does that differ between Whittle sizes?

The table-wide median IQ4_NL error is a flat ~8.05%, but sampling across the whole table also hits rows
with 100%+ relative error - rows whose values are so small that the fp16 block scale cannot represent
them.  Head 0 has almost none of those; sampling the whole table finds plenty.  So the zero/near-zero
rows are NOT uniformly sprinkled: they live in particular heads.

This matters for the size argument: an all-zero row costs the same 144 B in IQ4_NL and 272 B in Q8_0, so
if a large share of heads are empty, the table is mostly padding and the "10 B memory" is not 10 B of
knowledge.
"""
import struct
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths  # noqa: E402

_paths.add_gguf_py()
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from ple_iq4_pack import find_table, row_bytes  # noqa: E402

HEADS = 8


def head_stats(path, per_head=600):
    ds, (dims, tt, off) = find_table(pathlib.Path(path))
    head_dim, total = int(dims[0]), int(dims[1])
    rb = row_bytes(head_dim, tt)
    rows_per_head = total // HEADS
    rng = np.random.default_rng(3)
    print("%s" % pathlib.Path(path).name)
    print("  %d rows x %d values, %s, %d B/row" % (total, head_dim, QT(tt).name, rb))
    print("  %-6s %-8s %10s %10s %10s" % ("head", "kind", "zero%", "tiny%", "medianRMS"))
    for h in range(HEADS):
        base = h * rows_per_head
        idx = np.sort(rng.integers(0, rows_per_head, per_head)) + base
        vals = np.empty((per_head, head_dim), dtype=np.float32)
        with open(path, "rb") as f:
            for i, r in enumerate(idx):
                f.seek(ds + off + int(r) * rb)
                vals[i] = np.asarray(dequantize(np.frombuffer(f.read(rb), dtype=np.uint8), QT(tt)),
                                     dtype=np.float32).reshape(head_dim)
        rms = np.sqrt((vals ** 2).mean(axis=1))
        zero = 100 * float((rms == 0).mean())
        tiny = 100 * float((rms < 1e-3).mean())
        kind = "bigram" if h < HEADS // 2 else "trigram"
        print("  %-6d %-8s %9.1f%% %9.1f%% %10.5f" % (h, kind, zero, tiny, float(np.median(rms))))
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        head_stats(p)
