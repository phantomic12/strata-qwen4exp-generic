#!/usr/bin/env python3
"""Cross-check: does the IQ4_NL block this tool writes decode the same under my decoder and gguf-py's?

The pack tool reported 27-35% relative error where a prior measurement of the same encoding gave 8.67%.
One of the two decoders is reading the block differently, so decode the SAME bytes both ways and report
the disagreement.  gguf-py is llama.cpp's own code, so it is the arbiter.
"""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths  # noqa: E402

_paths.add_gguf_py()
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ple_iq4_pack import encode_iq4_nl, KV  # noqa: E402


def my_decode(blocks):
    """The layout the pack tool assumes: fp16 d, low nibble = element j, high nibble = element j+16."""
    out = np.empty((blocks.shape[0], 32), dtype=np.float32)
    for i in range(blocks.shape[0]):
        b = blocks[i]
        d = np.frombuffer(b[:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        qs = b[2:]
        out[i, :16] = KV[qs & 0x0F] * d
        out[i, 16:] = KV[qs >> 4] * d
    return out


rng = np.random.default_rng(0)
n = 4000
vals = (rng.standard_normal((n, 32)).astype(np.float32) * 0.004)
enc = encode_iq4_nl(vals)                       # n x 144 -> 8 blocks per row of 256; use 32-wide input

mine = my_decode(enc)
theirs = np.asarray(dequantize(enc.reshape(-1), QT.IQ4_NL), dtype=np.float32).reshape(n, 32)

print("blocks: %d, row bytes %d" % (n, enc.shape[1]))
print()
for label, back in (("mine", mine), ("gguf-py", theirs)):
    d = back - vals
    rmse = float(np.sqrt((d ** 2).mean()))
    rel = 100 * rmse / float(np.sqrt((vals ** 2).mean()))
    print("  %-8s rel rmse %7.3f%%  max|err| %.6f" % (label, rel, float(np.abs(d).max())))
print()
diff = mine - theirs
print("  mine vs gguf-py: max |diff| = %.6f, identical = %s"
      % (float(np.abs(diff).max()), bool(np.array_equal(mine, theirs))))
if not np.array_equal(mine, theirs):
    i = int(np.argmax(np.abs(diff).sum(axis=1)))
    print("  worst block %d" % i)
    print("    vals   :", np.array2string(vals[i][:8], precision=5))
    print("    mine   :", np.array2string(mine[i][:8], precision=5))
    print("    gguf-py:", np.array2string(theirs[i][:8], precision=5))
    print("    scale  :", np.frombuffer(enc[i][:2].tobytes(), dtype=np.float16)[0])
    print("    qs     :", enc[i][2:10])
    # hypothesis: gguf-py treats the two nibbles the other way round, or uses a different codebook
    swapped = np.empty_like(theirs)
    swapped[:, :16] = KV[enc[i][2:] >> 4] * np.frombuffer(enc[i][:2].tobytes(), dtype=np.float16)[0]
    swapped[:, 16:] = KV[enc[i][2:] & 0x0F] * np.frombuffer(enc[i][:2].tobytes(), dtype=np.float16)[0]
    print("    swapped-nibble decode of this block:", np.array2string(swapped[i][:8], precision=5))
