#!/usr/bin/env python3
"""Sequential rows vs random rows: how much does IQ4_NL engram error depend on which rows you look at?

The pack tool verified the FIRST 200k rows and saw 27-35% relative error, where sampling random rows
across the whole table gave 8.67%.  The encoder is byte-identical to gguf-py's decoder (ple_iq4_xcheck),
so the difference has to be row-dependent.  This measures both populations through one code path.
"""
import random
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths  # noqa: E402

_paths.add_gguf_py()
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from ple_iq4_pack import encode_iq4_nl, find_table, row_bytes  # noqa: E402


def measure(path, rows, n):
    ds, (dims, tt, off) = find_table(pathlib.Path(path))
    head_dim, total = int(dims[0]), int(dims[1])
    rb = row_bytes(head_dim, tt)
    out = np.empty((len(rows), head_dim), dtype=np.float32)
    with open(path, "rb") as f:
        for i, r in enumerate(rows):
            f.seek(ds + off + r * rb)
            raw = f.read(rb)
            out[i] = np.asarray(dequantize(np.frombuffer(raw, dtype=np.uint8), QT(tt)),
                                dtype=np.float32).reshape(head_dim)
    enc = encode_iq4_nl(out)
    back = np.asarray(dequantize(enc.reshape(-1), QT.IQ4_NL), dtype=np.float32).reshape(out.shape)
    # per-row relative error, then summarised: a table-wide "relative error" is a mean of a ratio and
    # rows with small RMS dominate it, so report the distribution not just the aggregate
    num = np.sqrt(((back - out) ** 2).mean(axis=1))
    den = np.sqrt((out ** 2).mean(axis=1))
    rel = np.where(den > 0, 100 * num / np.maximum(den, 1e-30), 0.0)
    return rel, den


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path.home() / "models/w45_Q8_0.gguf")
    n = 3000
    rng = random.Random(5)
    ds, (dims, tt, off) = find_table(pathlib.Path(path))
    total = int(dims[1])

    for label, rows in (("rows 0..%d (sequential)" % (n - 1), list(range(n))),
                        ("random rows, whole table", sorted(rng.sample(range(total), n))),
                        ("random rows, head 0 only", sorted(rng.sample(range(0, total // 8), n)))):
        rel, den = measure(path, rows, n)
        print("%s" % label)
        print("  rel err  median %6.2f%%  mean %6.2f%%  p90 %6.2f%%  max %6.2f%%"
              % (np.median(rel), rel.mean(), np.percentile(rel, 90), rel.max()))
        print("  row RMS  median %.5f   (rows with RMS < 1e-3: %d/%d)"
              % (np.median(den), int((den < 1e-3).sum()), n))
        print()


if __name__ == "__main__":
    main()
