"""Stage 0: exact per-head statistics from the base bf16 table itself.

Three separable reasons a head can contribute little, and they need different fixes:

  (a) the row is EMPTY                -> nothing to gate. Capacity/coverage problem.
  (b) the row is live but QUIET        -> tiny value norm. A scale problem.
  (c) value is fine, GATE closed       -> key/query disagree. A training-signal problem.

This measures (a) and (b) exactly, from the checkpoint's own bf16 tensors over HTTP range — not from the
Q8_0 GGUF, and not from a reimplementation. (c) needs real layer-2 hidden states, so it is stage 1 on a
GPU; this file says so rather than faking it.

Shard layout (verified from the index): shard_0 rows 0-7,812,499; shard_1 7,812,500-15,624,999;
shard_2 15,625,000-23,437,499; shard_3 23,437,500-31,249,999; shard_4 31,250,000-39,039,999 (7,790,000).
"""
import argparse
import json
import struct
import sys
import urllib.request

import numpy as np

REPO = "whittleOrg/Whittle-Qwen-3.8-45B-A3B"
BASE = "https://huggingface.co/%s/resolve/main/" % REPO
HEADS, HEAD_DIM = 8, 256
OFFSETS = [4_880_000 * i for i in range(HEADS)]
VOCAB = [4_880_000] * HEADS

# [(shard file, first row, n rows)]
SHARDS = [
    ("model-00012.safetensors", 0, 7_812_500),
    ("model-00013.safetensors", 7_812_500, 7_812_500),
    ("model-00013.safetensors", 15_625_000, 7_812_500),
    ("model-00014.safetensors", 23_437_500, 7_812_500),
    ("model-00014.safetensors", 31_250_000, 7_790_000),
]


def get(url, start=None, end=None, retries=4):
    for k in range(retries):
        try:
            req = urllib.request.Request(url)
            if start is not None:
                req.add_header("Range", "bytes=%d-%d" % (start, end))
            return urllib.request.urlopen(req, timeout=180).read()
        except Exception:
            if k == retries - 1:
                raise
    raise RuntimeError


_hcache = {}


def header(fname):
    if fname in _hcache:
        return _hcache[fname]
    n, = struct.unpack("<Q", get(BASE + fname, 0, 7))
    hdr, ds = json.loads(get(BASE + fname, 8, 8 + n - 1)), 8 + n
    _hcache[fname] = (hdr, ds)
    return hdr, ds


def shard_for(row):
    for i, (fn, first, n) in enumerate(SHARDS):
        if first <= row < first + n:
            return i, fn, row - first
    raise IndexError(row)


def tensor_rows(fname, name, local_rows, ncol=HEAD_DIM, batch=256):
    """Read selected rows (local to that shard) of a bf16 [nrow, ncol] tensor."""
    hdr, ds = header(fname)
    b0, _ = hdr[name]["data_offsets"]
    rowbytes = ncol * 2
    out = np.empty((len(local_rows), ncol), dtype=np.float32)
    for i in range(0, len(local_rows), batch):
        chunk = local_rows[i:i + batch]
        for j, r in enumerate(chunk):
            a = ds + b0 + int(r) * rowbytes
            raw = get(BASE + fname, a, a + rowbytes - 1)
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            out[i + j] = (u16 << 16).view(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="data/tokens_chat.npy")
    ap.add_argument("--n-tokens", type=int, default=4000)
    ap.add_argument("--per-head", type=int, default=120)
    a = ap.parse_args()

    sys.path.insert(0, ".")
    from hash import ngram_rows

    mult = [32473892291597, 22010589867355, 6403636407469]
    tok = np.load(a.tokens).astype(np.int64).ravel()[: a.n_tokens]
    rows = ngram_rows(tok, mult, VOCAB, OFFSETS, 3, 4, 248044).astype(np.int64)
    rng = np.random.default_rng(0)
    print("tokens %d -> %d hashed rows per head; sampling %d per head from the real table"
          % (len(tok), len(rows), a.per_head))

    print("\n  %-5s %-7s %11s %11s %11s %9s %9s"
          % ("head", "kind", "mean|row|", "median|row|", "p90|row|", "empty", "live"))
    summary = []
    for h in range(HEADS):
        rid = rng.choice(rows[:, h], size=min(a.per_head, len(rows)), replace=False)
        # group reads by shard so each fetch is contiguous-ish
        acc = []
        for shard_idx, (fn, first, nrows) in enumerate(SHARDS):
            sel = rid[(rid >= first) & (rid < first + nrows)]
            if not len(sel):
                continue
            name = "model.ngram_embedding.shard_%d.weight" % shard_idx
            acc.append(tensor_rows(fn, name, sel - first))
        g = np.concatenate(acc, axis=0)
        nrm = np.linalg.norm(g, axis=1)
        empty = float((nrm == 0).mean())
        live = nrm[nrm > 0]
        kind = "bigram" if h < 4 else "trigram"
        med = float(np.median(live)) if len(live) else 0.0
        p90 = float(np.percentile(live, 90)) if len(live) else 0.0
        mean = float(live.mean()) if len(live) else 0.0
        summary.append(dict(head=h, kind=kind, mean=mean, median=med, p90=p90,
                            empty=empty, live=1.0 - empty))
        print("  %-5d %-7s %11.4f %11.4f %11.4f %8.1f%% %8.1f%%"
              % (h, kind, mean, med, p90, 100 * empty, 100 * (1 - empty)))

    json.dump(summary, open("data/stage0_head_stats.json", "w"), indent=1)

    print("\n-- ratio of live-row magnitude to the bigram heads (0-2 pooled) --")
    ref = np.mean([s["median"] for s in summary if s["head"] in (0, 1, 2)])
    for s in summary:
        print("  head %d (%s): %.2fx" % (s["head"], s["kind"], s["median"] / ref if ref else 0))

    print("\nGATE magnitudes are NOT measured here: they need real layer-2 hidden states.")
    print("That is stage 1 on a GPU, and stage0_head_stats.json is its baseline.")


if __name__ == "__main__":
    main()
