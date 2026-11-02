#!/usr/bin/env python3
"""tools/ple_iq4_pack.py - rewrite a qwen4exp GGUF's engram table as IQ4_NL, so any size becomes runnable.

The engine reads `per_layer_token_embd.weight` out of the *original* GGUF (it is not in the canonical
pack), and the table's type is whatever llama-quantize gave the file.  Whittle ships four sizes and the
table follows the file: Q3_K / Q4_K / Q5_K / Q8_0.  The engine has no K-quant row decoder, so three of
the four stop at "per_layer_token_embd.weight is Q4_K, not <list>".

IQ4_NL is the format this engine reads best for the table: 18 B per 32 values, a 16-entry non-linear
codebook that spends its resolution near zero where these weights live, and `ple_formats()` already
carries both its decoder and its block form.  Converting the table to IQ4_NL therefore:

  * makes Q3_K_M / Q4_K_M / Q5_K_M loadable at all (the blocker is the table's type, not the experts);
  * makes the table 144 B/row whatever size the body is, so the PLE term stops growing with the quant
    (272 B at Q8_0, 144-176 B for the K-quants);
  * costs ~8.7% relative weight error against the Q8_0 table, measured by tools/engram_quant_error.py,
    against ~9.4% for plain Q4_0 at the same row size - the codebook earns ~8%, it is not free.

Writes a sidecar GGUF holding only the table, passed as `--ple-gguf`; the main model file is untouched,
so a size that already works keeps working:

    python tools/ple_iq4_pack.py --gguf Whittle-Q4_K_M.gguf --out w45-engram-iq4nl.gguf

WHY THIS ENCODES THE BLOCKS ITSELF.  gguf-py decodes IQ4_NL but raises NotImplementedError on
`quantize(..., IQ4_NL)`, and `llama-quantize` refuses the file outright ("unknown model architecture:
'qwen4exp'").  So the encoder is here.  It is a reference encoder, not a transcription of llama.cpp's:
for each 32-value block it tries several candidate fp16 scales and keeps the one with the least
round-trip squared error, then maps each weight to the nearest codebook entry.  That considers a
superset of what llama.cpp's single fixed divisor does, so the blocks are never worse.  `--verify`
decodes the written bytes back with gguf-py's own IQ4_NL decoder - the authoritative one - and prints
the error, so a wrong block layout fails loudly instead of shipping.
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths  # noqa: E402

_paths.add_gguf_py()
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

GGUF_TYPE_STRING = 8
GGML_TYPE_IQ4_NL = 20
ALIGN = 32
CHUNK_ROWS = 4096

# ggml-common.h kvalues_iq4nl: the 16-entry non-linear codebook, signed
KV = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
               1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key: str, val: str) -> bytes:
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + gguf_string(val)


def find_table(path: pathlib.Path):
    """(data_start, dims, ggml type, tensor offset) for per_layer_token_embd.weight."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit("%s is not a GGUF" % path)
        struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def rs():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(t):
            if t == 8:
                rs()
            elif t == 9:
                (et,) = struct.unpack("<I", f.read(4))
                (c,) = struct.unpack("<Q", f.read(8))
                for _ in range(c):
                    skip(et)
            else:
                f.read(FIXED[t])

        for _ in range(n_kv):
            rs()
            (vt,) = struct.unpack("<I", f.read(4))
            skip(vt)
        found = None
        for _ in range(n_tensors):
            name = rs()
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            (tt,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            if name == "per_layer_token_embd.weight":
                found = (dims, tt, off)
        pos = f.tell()
    if found is None:
        sys.exit("%s has no per_layer_token_embd.weight" % path)
    return (pos + ALIGN - 1) // ALIGN * ALIGN, found


def row_bytes(head_dim: int, qtype: int):
    try:
        bs, ts = GGML_QUANT_SIZES[QT(qtype)]
    except KeyError:
        return None
    if head_dim % bs:
        return None
    return (head_dim // bs) * ts


def encode_iq4_nl(values: np.ndarray) -> np.ndarray:
    """(rows, head_dim) f32 -> IQ4_NL bytes, 18 per 32 values.

    Block layout (ggml's block_iq4_nl): fp16 scale `d`, then 16 bytes whose LOW nibble indexes element
    j and whose HIGH nibble indexes element j+16.  The codebook carries its own signs, so an index
    selects a signed value and a negative weight is a codebook entry, not a sign bit.
    """
    rows, dim = values.shape
    nblk = dim // 32
    blocks = values.reshape(rows * nblk, 32).astype(np.float32)
    absmax = np.abs(blocks).max(axis=1)
    safe = np.where(absmax == 0.0, 1.0, absmax).astype(np.float32)

    # Candidate scales: put the block's largest magnitude onto each of the largest codebook entries.
    # Each entry gives a different grid over the block; the best is kept by round-trip squared error.
    cand = np.abs(KV)[np.argsort(-np.abs(KV))][:6]          # 127, 113, 104, 89, 83, 69
    best_d = np.zeros(blocks.shape[0], dtype=np.float32)
    best_err = np.full(blocks.shape[0], np.inf, dtype=np.float32)

    for c in cand:
        d = (safe / c).astype(np.float32)
        q = blocks / d[:, None]
        idx = np.abs(q[:, :, None] - KV[None, None, :]).argmin(axis=2)
        rec = KV[idx] * d[:, None]
        err = ((rec - blocks) ** 2).sum(axis=1)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)

    # The scale is stored as fp16, so quantise against the value that will actually be written
    d16 = best_d.astype(np.float16).astype(np.float32)
    q = blocks / np.where(d16 == 0.0, np.float32(1.0), d16)[:, None]
    idx = np.abs(q[:, :, None] - KV[None, None, :]).argmin(axis=2).astype(np.uint8)

    out = np.empty((rows * nblk, 18), dtype=np.uint8)
    out[:, :2] = np.frombuffer(d16.astype(np.float16).tobytes(), dtype=np.uint8).reshape(-1, 2)
    out[:, 2:] = (idx[:, :16] & 0x0F) | ((idx[:, 16:] & 0x0F) << 4)
    return out.reshape(rows, nblk * 18)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True, help="the qwen4exp GGUF to take the table from")
    ap.add_argument("--out", required=True, help="the sidecar GGUF to write (table only, IQ4_NL)")
    ap.add_argument("--rows", type=int, default=0, help="debug: only the first N rows")
    ap.add_argument("--force", action="store_true", help="rewrite even if the table is already IQ4_NL")
    ap.add_argument("--verify", type=int, default=200, help="decode this many written rows back and report error")
    a = ap.parse_args()

    src, out = pathlib.Path(a.gguf), pathlib.Path(a.out)
    ds, (dims, tt, off) = find_table(src)
    if len(dims) != 2:
        sys.exit("per_layer_token_embd.weight is %d-D; expected [head_dim, rows]" % len(dims))
    head_dim, n_rows = int(dims[0]), int(dims[1])
    src_type = QT(tt).name
    print("source : %s" % src)
    print("table  : %d rows x %d values, %s" % (n_rows, head_dim, src_type))
    if src_type == "IQ4_NL" and not a.force:
        print("already IQ4_NL - nothing to do (--force to rewrite anyway)")
        return
    if head_dim % 32:
        sys.exit("head_dim %d is not a multiple of 32; IQ4_NL blocks are 32 values" % head_dim)

    rbin = row_bytes(head_dim, tt)
    if rbin is None:
        sys.exit("cannot determine the byte width of a %s row at head_dim %d" % (src_type, head_dim))
    rout = 18 * (head_dim // 32)
    print("row    : %d B in, %d B out (18 x %d blocks)" % (rbin, rout, head_dim // 32))

    if a.rows:
        n_rows = min(n_rows, a.rows)
    out_bytes = rout * n_rows
    print("output : %d rows, %.2f GB" % (n_rows, out_bytes / 1e9))

    kvs = [kv_string("general.architecture", "strata-ple"),
           kv_string("general.name", "PLE n-gram table, IQ4_NL (repacked from %s)" % src_type),
           kv_string("strata.ple.format", "iq4_nl"),
           kv_string("strata.ple.source", src.name),
           kv_string("strata.ple.source_type", src_type)]
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs)
    head += gguf_string("per_layer_token_embd.weight") + struct.pack("<I", 2)
    head += struct.pack("<QQ", head_dim, n_rows)
    head += struct.pack("<I", GGML_TYPE_IQ4_NL) + struct.pack("<Q", 0)
    head += b"\0" * ((-len(head)) % ALIGN)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    n_verify = min(a.verify, n_rows)
    picks = np.linspace(0, n_rows - 1, n_verify).astype(int) if n_verify else np.zeros(0, dtype=int)
    verify_at = {int(v): k for k, v in enumerate(picks)}
    orig_s = np.zeros((n_verify, head_dim), dtype=np.float32)
    enc_s = np.zeros((n_verify, rout), dtype=np.uint8)

    done = 0
    with open(tmp, "wb") as w, open(src, "rb") as f:
        w.write(head)
        for start in range(0, n_rows, CHUNK_ROWS):
            cnt = min(CHUNK_ROWS, n_rows - start)
            f.seek(ds + off + start * rbin)
            raw = f.read(cnt * rbin)
            if len(raw) != cnt * rbin:
                sys.exit("short read at row %d" % start)
            vals = np.asarray(dequantize(np.frombuffer(raw, dtype=np.uint8), QT(tt)),
                              dtype=np.float32).reshape(cnt, head_dim)
            enc = encode_iq4_nl(vals)
            w.write(enc.tobytes())
            for r in range(cnt):
                k = verify_at.get(start + r)
                if k is not None:
                    orig_s[k] = vals[r]
                    enc_s[k] = enc[r]
            done += cnt
            print("  %d/%d rows" % (done, n_rows), flush=True)

    expect = len(head) + out_bytes
    got = tmp.stat().st_size
    if got != expect:
        sys.exit("size check failed: %d != %d" % (got, expect))
    tmp.replace(out)
    print("wrote %s (%d B header + %d B table = %.2f GB)" % (out, len(head), out_bytes, got / 1e9))

    if n_verify:
        # decode with gguf-py's own IQ4_NL decoder: the authoritative check that the written bytes are
        # a valid IQ4_NL tensor in ggml's own layout
        back = np.asarray(dequantize(enc_s.reshape(-1), QT.IQ4_NL), dtype=np.float32).reshape(orig_s.shape)
        d = back - orig_s
        rmse = float(np.sqrt((d ** 2).mean()))
        rel = 100 * rmse / (float(np.sqrt((orig_s ** 2).mean())) or 1.0)
        print("\nverify (gguf-py decode of %d written rows): rmse %.6f, relative %.3f%%, max|err| %.5f"
              % (n_verify, rmse, rel, float(np.abs(d).max())))
    print("\nrun with:  strata ... --ple-gguf %s" % out)


if __name__ == "__main__":
    main()
