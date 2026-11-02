"""Read the ngram buffer tensors straight out of the base safetensors header.

Whittle's GGUF says head_vocab_sizes = eight identical 4,880,000s. The published transformers
reference builds them as eight DISTINCT primes just above 4,880,000, which would make the total row
count 39,040,548 (padded 39,040,640) instead of 39,040,000. Only one can be what the weights were
trained with, and the base checkpoint stores both buffers explicitly:

    model.ple_embedding.ngram_heads_vocab_sizes
    model.ple_embedding.ngram_heads_offsets
    model.ple_embedding.layer_multipliers

This reads their actual values from the safetensors header + data section via HTTP range requests, so
the answer is the trained bytes rather than either implementation's opinion.
"""
import json
import struct
import sys
import urllib.request

REPO = "whittleOrg/Whittle-Qwen-3.8-45B-A3B"
BASE = "https://huggingface.co/%s/resolve/main/" % REPO

DTYPE_SIZE = {"I64": 8, "F64": 8, "I32": 4, "F32": 4, "I16": 2, "BF16": 2, "F16": 2}


def get(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", "bytes=%d-%d" % (start, end))
    return urllib.request.urlopen(req, timeout=120).read()


def header(fname):
    """safetensors: u64 header_len, then JSON."""
    raw = get(BASE + fname, 0, 7)
    n, = struct.unpack("<Q", raw)
    js = get(BASE + fname, 8, 8 + n - 1)
    return json.loads(js), 8 + n


def read_tensor(fname, name):
    hdr, data_start = header(fname)
    if name not in hdr:
        return None, None
    info = hdr[name]
    dt = info["dtype"]
    shape = info["shape"]
    b0, b1 = info["data_offsets"]
    isz = DTYPE_SIZE[dt]
    raw = get(BASE + fname, data_start + b0, data_start + b1 - 1)
    fmt = {"I64": "<q", "F64": "<d", "I32": "<i", "F32": "<f", "I16": "<h"}.get(dt)
    count = len(raw) // isz
    vals = list(struct.unpack("<%d%s" % (count, fmt[1]), raw)) if fmt else None
    return shape, vals


def main():
    hdr, _ = header("model-00012.safetensors")
    print("tensors in shard 12 containing 'ple' or 'ngram':")
    for k in sorted(hdr):
        if "ple" in k.lower() or "ngram" in k.lower():
            print("  %-46s %-6s %s" % (k, hdr[k]["dtype"], hdr[k]["shape"]))

    print()
    for name in ("model.ple_embedding.ngram_heads_vocab_sizes",
                 "model.ple_embedding.ngram_heads_offsets",
                 "model.ple_embedding.layer_multipliers"):
        shape, vals = read_tensor("model-00012.safetensors", name)
        if shape is None:
            print("%s: ABSENT" % name)
            continue
        print("%s\n  shape %s\n  values %s" % (name, shape, vals))
        if "vocab_sizes" in name and vals:
            print("  distinct?      %s" % (len(set(vals)) == len(vals)))
            print("  all equal?     %s" % (len(set(vals)) == 1))
            print("  sum            %d" % sum(vals))
            print("  == 8*4880000   %s" % (sum(vals) == 8 * 4_880_000))

    # and the embedding's real row count, from the shard shapes
    print("\nngram shard shapes (rows x dim):")
    total_rows = 0
    for i in range(5):
        fn = "model-00012.safetensors" if i == 0 else (
             "model-00013.safetensors" if i in (1, 2) else "model-00014.safetensors")
        hdr, _ = header(fn)
        k = "model.ngram_embedding.shard_%d.weight" % i
        if k in hdr:
            sh = hdr[k]["shape"]
            total_rows += sh[0]
            print("  shard_%d  %-6s %s" % (i, hdr[k]["dtype"], sh))
    print("  total rows across shards: %d" % total_rows)


if __name__ == "__main__":
    main()
