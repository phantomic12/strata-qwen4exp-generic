"""Which table is real: the checkpoint's own bytes, or what transformers rebuilds?

Stage 1b showed the loaded module uses
  head_vocab_sizes [4880011, 4880021, ...]      (distinct primes)
  multipliers     [23703573157769, ...]
  weight          (39,040,640, 256)
while the checkpoint's stored buffers and the GGUF both use
  head_vocab_sizes [4880000] * 8                (equal)
  multipliers     [32473892291597, ...]
  rows            39,040,000

Those are different models. This compares the two head to head, on the raw safetensors and on the
loaded module, so there is no doubt which one the weights were trained with:

  A) occupancy per head, computed from the RAW shard files (ground truth, no module involved),
     under both the equal-size layout and the prime layout
  B) whether the loaded module's rows equal the raw shard's rows (i.e. did loading preserve them)
"""
import json
import math

import modal

app = modal.App("whittle-engram-layout")

HF_CACHE = modal.Volume.from_name("whittle-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformers==5.22.1", "accelerate", "safetensors",
                 "huggingface_hub", "numpy", "tokenizers", "datasets")
    .env({"HF_HOME": "/cache"})
)

REPO = "whittleOrg/Whittle-Qwen-3.8-45B-A3B"


def primes_after(base, n):
    def isp(v):
        if v < 2:
            return False
        if v % 2 == 0:
            return v == 2
        for d in range(3, math.isqrt(v) + 1, 2):
            if v % d == 0:
                return False
        return True
    out, p = [], base
    for _ in range(n):
        p += 1
        while not isp(p):
            p += 1
        out.append(p)
    return out


@app.function(image=image, gpu="H200", volumes={"/cache": HF_CACHE}, timeout=3600)
def check():
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    path = snapshot_download(REPO, allow_patterns=["*.safetensors", "*.json"])
    files = ["model-00012.safetensors", "model-00013.safetensors", "model-00014.safetensors"]
    prefix = path + "/"

    # --- A) raw table, straight from the files ---
    raw = {}
    for i, f in enumerate(files):
        with safe_open(prefix + f, framework="pt", device="cpu") as fh:
            for key in fh.keys():
                if key.startswith("model.ngram_embedding.shard_"):
                    raw[key] = fh.get_tensor(key)
    print("raw shards:", {k.split(".")[-2]: tuple(v.shape) for k, v in raw.items()})
    table = torch.cat([raw["model.ngram_embedding.shard_%d.weight" % i] for i in range(5)], dim=0)
    print("concatenated raw table:", tuple(table.shape), table.dtype)

    def occupancy(rows_per_head, offsets, sample=3000):
        out = []
        W = table.float()
        for h, (lo, n) in enumerate(zip(offsets, rows_per_head)):
            idx = torch.arange(lo, lo + n, max(1, n // sample))
            nrm = W[idx].norm(dim=1)
            out.append(float((nrm > 0).float().mean()))
        return out

    eq_sizes = [4_880_000] * 8
    eq_offs = [4_880_000 * i for i in range(8)]
    pr = primes_after(4_879_999, 8)
    pr_offs, acc = [], 0
    for s in pr:
        pr_offs.append(acc)
        acc += s

    print("\n--- occupancy from RAW bytes ---")
    print("equal-size layout (39,040,000 rows):", [round(100 * v, 1) for v in occupancy(eq_sizes, eq_offs)])
    if acc <= table.shape[0]:
        print("prime layout      (%d rows):" % acc, [round(100 * v, 1) for v in occupancy(pr, pr_offs)])

    # --- B) did the module preserve the rows? ---
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, device_map="auto")
    ple = None
    for name, m in model.named_modules():
        if type(m).__name__ == "Qwen4ExpTextPLELayer":
            ple = m
    W = ple.ple_embedding.ngram_embedding.weight
    print("\nmodule embedding:", tuple(W.shape))

    n = min(20000, table.shape[0], W.shape[0])
    a = table[:n].float()
    b = W[:n].float().cpu()
    same_rows = torch.allclose(a, b, atol=1e-2)
    print("first %d rows identical? %s" % (n, same_rows))
    nz_raw = int((a.norm(dim=1) > 0).sum())
    nz_mod = int((b.norm(dim=1) > 0).sum())
    print("non-empty rows in first %d: raw %d vs module %d" % (n, nz_raw, nz_mod))

    # and the live count over a strided slice of the WHOLE module table
    idx = torch.arange(0, W.shape[0], 20011, device=W.device)
    nrm = W[idx].float().norm(dim=1)
    print("module occupancy over %d sampled rows: %.1f%% live"
          % (len(idx), 100 * float((nrm > 0).float().mean())))
    idx2 = torch.arange(0, table.shape[0], 20011)
    nrm2 = table[idx2].float().norm(dim=1)
    print("raw    occupancy over %d sampled rows: %.1f%% live"
          % (len(idx2), 100 * float((nrm2 > 0).float().mean())))

    return {
        "raw_shape": list(table.shape), "module_shape": list(W.shape),
        "rows_identical": bool(same_rows),
        "occupancy_equal_layout": occupancy(eq_sizes, eq_offs),
        "occupancy_prime_layout": occupancy(pr, pr_offs) if acc <= table.shape[0] else None,
        "module_live_frac": float((nrm > 0).float().mean()),
        "raw_live_frac": float((nrm2 > 0).float().mean()),
    }


@app.local_entrypoint()
def main():
    print(json.dumps(check.remote(), indent=1))
