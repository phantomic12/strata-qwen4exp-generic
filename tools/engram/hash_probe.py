"""Register measurement: how often does each engram head land on a LIVE row, on real text?

The table is 8 heads x 4,880,000 rows. We measured the table's occupancy offline: head 3 is 80% zero,
head 7 66%, the rest nearly full. That says where the table has holes, not whether the model's token
stream ever lands in them.

This runs the real hash (bitwise_xor of token * layer_multiplier, reduced mod each head's vocab size,
plus the head's offset - exactly what Qwen4ExpTextNGramEmbedding.forward does) over a real corpus and
answers:

  * per-head hit rate on live rows, and how it compares to the table's own occupancy
  * how many DISTINCT rows each head actually touches (the working set - this is what a hot-row cache
    would need to hold, and what bounds how many rows a finetune could ever reach)
  * the share of gathers that read a zero row (pure waste: a page fault for a zero vector)

If a head's live-row hit rate matches its occupancy, the holes are just unseen n-grams and the model
routes around them. If a head's hit rate is much WORSE than its occupancy, the hash is actively steering
into holes - a different and worse problem, and one a finetune could not fix without re-addressing rows.

CPU only. Reads the table's fp16 scales a few thousand rows at a time, no dequant, no GPU.
"""
import argparse
import json
import struct
import sys
from collections import Counter

import numpy as np


def read_gguf(path):
    """(metadata scalars/arrays, tensor infos, data_start) for a GGUF - header block only."""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", "not a GGUF"
        ver, = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def rs():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        FIX = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        ARR = {0: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        meta = {}

        def val(t):
            if t == 8:
                return rs()
            if t == 9:
                et, = struct.unpack("<I", f.read(4))
                cnt, = struct.unpack("<Q", f.read(8))
                if et in ARR:
                    a = np.frombuffer(f.read(cnt * FIX[et]), dtype=np.dtype(ARR[et]))
                    return a.tolist()
                return [val(et) for _ in range(cnt)]
            fmt = {0: "<b", 1: "<B", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
            return struct.unpack(fmt, f.read(FIX[t]))[0]

        for _ in range(n_kv):
            k = rs()
            t, = struct.unpack("<I", f.read(4))
            meta[k] = val(t)

        infos = {}
        for _ in range(n_tensors):
            name = rs()
            nd, = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            tt, = struct.unpack("<I", f.read(4))
            off, = struct.unpack("<Q", f.read(8))
            infos[name] = (dims, tt, off)
        pos = f.tell()
    return meta, infos, (pos + 31) // 32 * 32
