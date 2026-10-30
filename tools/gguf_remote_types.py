#!/usr/bin/env python3
"""Read only the GGUF header of a remote file (HTTP range) and report the expert tensor types.

Usage: gguf_remote_types.py <hf-url> [max_bytes]

GGUF layout: magic, version, n_tensors, n_kv, then the KV block, then the tensor-info block.
The tensor-info block names every tensor and its type, so a few MB of range requests at the
head of the file is enough -- no need to download a 48 GB quant to ask what types it uses.
"""
import struct
import sys
import urllib.request

# ggml type ids we care about
NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}

SZ = {0: 4, 1: 2, 2: 18, 3: 20, 6: 22, 7: 24, 8: 34, 9: 36, 10: 84, 11: 110, 12: 144, 13: 176,
      14: 210, 15: 292, 16: 66, 17: 74, 18: 98, 19: 56, 20: 18, 21: 110, 22: 82, 23: 34,
      24: 2, 25: 4, 26: 8, 27: 16, 28: 8, 29: 56, 30: 2}


class Range:
    def __init__(self, url):
        self.url = url
        self.buf = b""
        self.base = 0

    def need(self, upto):
        """Ensure the buffer covers [base, upto)."""
        while self.base + len(self.buf) < upto:
            start = self.base + len(self.buf)
            req = urllib.request.Request(self.url, headers={"Range": "bytes=%d-%d" % (start, start + (8 << 20) - 1)})
            with urllib.request.urlopen(req, timeout=120) as r:
                chunk = r.read()
            if not chunk:
                raise EOFError("stream ended at %d" % start)
            self.buf += chunk

    def take(self, off, n):
        self.need(off + n)
        i = off - self.base
        return self.buf[i:i + n]


def rd_str(r, off):
    (n,) = struct.unpack("<Q", r.take(off, 8))
    s = r.take(off + 8, n)
    return s.decode("utf-8", "replace"), off + 8 + n


def skip_val(r, off, t):
    if t == 8:
        return rd_str(r, off)[1]
    if t == 9:  # array
        (et,) = struct.unpack("<I", r.take(off, 4))
        (n,) = struct.unpack("<Q", r.take(off + 4, 8))
        off += 12
        if et == 8:
            for _ in range(n):
                off = rd_str(r, off)[1]
            return off
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}.get(et)
        if sz is None:
            raise ValueError("array elem type %d" % et)
        return off + n * sz
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}.get(t)
    if sz is None:
        raise ValueError("kv type %d" % t)
    return off + sz


def main():
    url = sys.argv[1]
    r = Range(url)
    magic = r.take(0, 4)
    if magic != b"GGUF":
        print("not a GGUF:", magic)
        return 1
    (ver,) = struct.unpack("<I", r.take(4, 4))
    (n_t, n_kv) = struct.unpack("<QQ", r.take(8, 16))
    print("version %d, %d tensors, %d kv" % (ver, n_t, n_kv))
    off = 24
    for _ in range(n_kv):
        k, off = rd_str(r, off)
        (t,) = struct.unpack("<I", r.take(off, 4))
        off = skip_val(r, off + 4, t)
    print("tensor info at %d" % off)

    roles = {}
    others = {}
    for _ in range(n_t):
        name, off = rd_str(r, off)
        (nd,) = struct.unpack("<I", r.take(off, 4))
        off += 4
        dims = struct.unpack("<%dQ" % nd, r.take(off, 8 * nd))
        off += 8 * nd
        (ty,) = struct.unpack("<I", r.take(off, 4))
        off += 4 + 8
        tn = NAMES.get(ty, "T%d" % ty)
        if name.startswith("blk.") and "_exps.weight" in name:
            role = name.split("ffn_")[1].split("_exps")[0]
            roles.setdefault(role, {}).setdefault(tn, [0, dims])
            roles[role][tn][0] += 1
        else:
            others.setdefault(tn, 0)
            others[tn] += 1
    print("\n=== expert roles ===")
    for role in ("gate", "up", "down"):
        if role in roles:
            for tn, (cnt, dims) in sorted(roles[role].items()):
                print("  %-5s %-9s x%-3d  dims=%s" % (role, tn, cnt, dims))
    print("\n=== non-expert tensors ===")
    for tn, c in sorted(others.items(), key=lambda x: -x[1]):
        print("  %-9s x%d" % (tn, c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
