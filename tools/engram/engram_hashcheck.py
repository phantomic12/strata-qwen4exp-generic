"""Stage 1b: does MY hash land on the same rows the model actually gathers?

Stage 1a found all 8 heads with identical gather norms (0.32) and zero empty rows in a real forward
pass. My offline probes found heads 3/7 at 0.30x/0.21x with 30%/55% empty. Both cannot be true.

This run settles it by comparing, for the same input ids:
  A) the ngram_ids the MODEL computes (hook its embedding module)
  B) the ngram_ids MY hash computes (tools/engram/hash.py)
and then reading the per-head row norms of the table the MODEL holds in memory, over the rows it
actually gathers.

Whichever disagrees is the bug, and everything downstream depends on it: if my hash is wrong, every
"head 3 is sparse" conclusion in ENGRAM_ANATOMY.md is an artefact of reading the wrong rows.
"""
import json

import modal

app = modal.App("whittle-engram-hashcheck")

HF_CACHE = modal.Volume.from_name("whittle-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformers==5.22.1", "accelerate", "safetensors",
                 "huggingface_hub", "numpy", "tokenizers", "datasets")
    .env({"HF_HOME": "/cache"})
    .add_local_file("hash.py", "/root/hash.py")
)

REPO = "whittleOrg/Whittle-Qwen-3.8-45B-A3B"


@app.function(image=image, gpu="H200", volumes={"/cache": HF_CACHE}, timeout=3600)
def check(n_tokens: int = 512):
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REPO)
    ds = load_dataset("tatsu-lab/alpaca", split="train[:200]")
    text = "".join("User: %s\nAssistant: %s\n" % (r["instruction"], r["output"]) for r in ds)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=n_tokens).input_ids
    ids = enc.to("cuda")
    print("tokens:", ids.shape)

    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, device_map="auto",
                                                 low_cpu_mem_usage=True)
    model.eval()

    ple = None
    for name, m in model.named_modules():
        if type(m).__name__ == "Qwen4ExpTextPLELayer":
            ple = m
            print("PLE:", name)
    ng_emb = ple.ple_embedding

    # --- the table the model holds ---
    W = ng_emb.ngram_embedding.weight
    print("embedding weight:", tuple(W.shape), W.dtype, "device", W.device)
    def _lst(x):
        return list(x) if isinstance(x, (list, tuple)) else x.tolist()

    print("head_vocab_sizes:", _lst(ng_emb.head_vocab_sizes))
    print("head_offsets    :", _lst(ng_emb.head_offsets))
    print("layer_multipliers:", _lst(ng_emb.layer_multipliers))

    # --- (A) the model's own ngram ids ---
    captured = {}
    h1 = ng_emb.ngram_embedding.register_forward_pre_hook(
        lambda m, args: captured.__setitem__("ids", args[0].detach()))
    h2 = ng_emb.register_forward_hook(
        lambda m, args, out: captured.__setitem__("emb", out.detach()))
    with torch.no_grad():
        model(ids)
    h1.remove()
    h2.remove()

    mid = captured["ids"]
    print("\nmodel ngram_ids:", tuple(mid.shape))
    print("  min/max:", int(mid.min()), int(mid.max()))

    # --- (B) my hash ---
    import sys
    sys.path.insert(0, "/root")
    from hash import ngram_rows
    mult = [int(v) for v in _lst(ng_emb.layer_multipliers)]
    vocab = [int(v) for v in _lst(ng_emb.head_vocab_sizes)]
    offs = [int(v) for v in _lst(ng_emb.head_offsets)]
    mine = ngram_rows(ids[0].cpu().numpy().astype(np.int64), mult, vocab, offs,
                      3, 4, int(ng_emb.eos_token_id))
    print("my   ngram_ids:", mine.shape)

    mm = mid[0].cpu().numpy().astype(np.int64)
    same = (mm == mine).all()
    print("\n=== DO THEY MATCH? %s ===" % ("YES" if same else "NO"))
    if not same:
        diff = (mm != mine)
        print("  mismatched entries: %d / %d (%.2f%%)"
              % (diff.sum(), diff.size, 100 * diff.mean()))
        for h in range(8):
            d = diff[:, h]
            print("    head %d: %.1f%% mismatch" % (h, 100 * d.mean()))
        i = int(np.argwhere(diff)[0][0])
        hh = int(np.argwhere(diff)[0][1])
        print("  first mismatch token %d head %d: model %d vs mine %d" % (i, hh, mm[i, hh], mine[i, hh]))

    # --- per-head row norms over the rows the MODEL gathered ---
    print("\n=== per-head table stats over the MODEL's gathered rows ===")
    print("  %-5s %-8s %10s %10s %10s %8s" % ("head", "kind", "mean", "median", "p90", "empty"))
    stats = []
    for h in range(8):
        rid = torch.as_tensor(mm[:, h], device=W.device)
        rows = W[rid].float()
        nrm = rows.norm(dim=1).cpu().numpy()
        st = dict(head=h, mean=float(nrm.mean()), median=float(np.median(nrm)),
                  p90=float(np.percentile(nrm, 90)), empty=float((nrm == 0).mean()))
        stats.append(st)
        print("  %-5d %-8s %10.4f %10.4f %10.4f %7.1f%%"
              % (h, "bigram" if h < 4 else "trigram", st["mean"], st["median"], st["p90"], 100 * st["empty"]))

    # --- and the table's own per-head occupancy over ALL rows (strided) ---
    print("\n=== per-head occupancy over ALL rows (stride 4099, from the live weight) ===")
    occ = []
    for h in range(8):
        lo, n = offs[h], vocab[h]
        idx = torch.arange(lo, lo + n, 4099, device=W.device)
        nrm = W[idx].float().norm(dim=1)
        f = float((nrm > 0).float().mean())
        occ.append(f)
        print("  head %d: %.1f%% live" % (h, 100 * f))

    # --- what the embedding actually returned, reshaped the way I assumed ---
    emb = captured["emb"].float()
    print("\n=== gathered embedding, reshaped [-1, 8, 256] ===")
    e = emb.reshape(-1, 8, 256)
    for h in range(8):
        nrm = e[:, h, :].norm(dim=1)
        print("  head %d: mean %.4f  median %.4f" % (h, float(nrm.mean()), float(nrm.median())))

    return {"match": bool(same), "stats_model_rows": stats,
            "occupancy_all_rows": occ,
            "head_vocab_sizes": vocab, "head_offsets": offs, "multipliers": mult}


@app.local_entrypoint()
def main(n_tokens: int = 512):
    r = check.remote(n_tokens=n_tokens)
    print("\n=== RESULT ===")
    print(json.dumps(r, indent=1))
