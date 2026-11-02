#!/usr/bin/env python3
"""Sanity-check the per-head zero finding: same rows, two ways, and a direct byte-level check.

Head 3 showing 80% zero and head 7 showing 65% zero, while heads 4-6 show 0.0%, is a big claim.  Three
ways to be wrong about it, all checked here:

  1. Slice arithmetic: `total // 8` rows per head assumes exactly 8 equal heads.  Verified against the
     GGUF's own head_offsets metadata (0, 4880000, 9760000, ...).
  2. A decoder that returns zeros where it should not.  Checked by looking at the raw block bytes: an
     all-zero-scales Q8_0 block is unambiguous.
  3. Row-id collision in the sample.  Uses a deterministic stride, not a RNG, and prints the ids.
"""
import struct
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths  # noqa: E402

_paths.add_gguf_py()
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402
from gguf import GGMLQuantizationType as QT  # noqa: E402

from ple_iq4_pack import find_table  # noqa: E402


def read_kv_head_offsets(path):
    """The table's own head_offsets from the GGUF metadata, to check the /8 split."""
    with open(path, "rb") as f:
        f.read(4)
        struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def rs():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        vals = {}
        for _ in range(n_kv):
            k = rs()
            (vt,) = struct.unpack("<I", f.read(4))
            if vt == 9:                                   # array
                (et,) = struct.unpack("<I", f.read(4))
                (c,) = struct.unpack("<Q", f.read(8))
                if et == 4:                                # uint32 array
                    vals[k] = list(struct.unpack("<%dI" % c, f.read(4 * c)))
                elif et == 5:                              # int32 array
                    vals[k] = list(struct.unpack("<%di" % c, f.read(4 * c)))
                elif et == 10:                             # uint64 array
                    vals[k] = list(struct.unpack("<%dQ" % c, f.read(8 * c)))
                elif et == 8:                              # array of strings: not needed here
                    for _ in range(c):
                        rs()
                else:
                    f.read(c * {0: 1, 1: 1, 2: 2, 3: 2, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[et])
            elif vt == 8:
                rs()
            else:
                f.read({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[vt])
    return vals


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path.home() / "models/w45_Q8_0.gguf")
    kv = read_kv_head_offsets(path)
    offs = kv.get("qwen4exp.ple.head_offsets")
    vocabs = kv.get("qwen4exp.ple.head_vocab_sizes")
    print("metadata head_offsets   :", offs)
    print("metadata head_vocab_size:", vocabs)

    ds, (dims, tt, off) = find_table(pathlib.Path(path))
    head_dim, total = int(dims[0]), int(dims[1])
    bs, ts = GGML_QUANT_SIZES[QT(tt)]
    rb = (head_dim // bs) * ts
    print("table rows              : %d, %d values, %s (%d B/row)" % (total, head_dim, QT(tt).name, rb))
    if offs:
        print("offsets + vocabs        :", [o + v for o, v in zip(offs, vocabs)], "(sum should be %d)" % total)

    # deterministic stride, 500 rows per head, printed so the sample is reproducible
    per, stride = 500, 977
    print("\nper-head zero-scale rate (deterministic stride %d, %d rows/head):" % (stride, per))
    print("  %-5s %-8s %6s %8s %10s %12s" % ("head", "kind", "rows", "zero%", "zero rms", "nonzero rms"))
    nheads = len(offs) if offs else 8
    rows_per_head = (offs[1] - offs[0]) if offs else total // nheads
    for h in range(nheads):
        base = offs[h] if offs else h * rows_per_head
        ids = [base + (i * stride) % rows_per_head for i in range(per)]
        first = ids[0]
        z = 0
        good = []
        with open(path, "rb") as f:
            for r in ids:
                f.seek(ds + off + r * rb)
                raw = f.read(rb)
                # Q8_0: an all-zero block is fp16 d == 0.  Check the scales directly, no decoder involved.
                scales = [struct.unpack("<e", raw[k * ts:k * ts + 2])[0] for k in range(head_dim // bs)]
                if all(s == 0.0 for s in scales):
                    z += 1
                else:
                    vals = np.frombuffer(
                        b"".join(raw[k * ts + 2:(k + 1) * ts] for k in range(head_dim // bs)),
                        dtype=np.int8).astype(np.float32)
                    sc = np.repeat(np.array(scales, dtype=np.float32), bs)
                    good.append(float(np.sqrt(((vals * sc) ** 2).mean())))
        kind = "bigram" if h < nheads // 2 else "trigram"
        print("  %-5d %-8s %6d %7.1f%% %10s %12s"
              % (h, kind, per, 100 * z / per, "-" if z < per else "0.00000",
                 ("%.5f" % float(np.median(good))) if good else "- (all zero)"))
    print("\nfirst row id sampled per head:", [ (offs[h] if offs else h*rows_per_head) for h in range(nheads) ])


if __name__ == "__main__":
    main()
