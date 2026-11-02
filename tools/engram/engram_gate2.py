"""Stage 1a (redo): the gate, on a table that is actually the trained one.

The first run of this measurement was void: from_pretrained left the engram embedding at its random
init (see engram_load.py for the proof), so it measured a random table. This version loads through
engram_load.load_whittle(), which places the real rows and refuses to continue unless the per-head
occupancy reproduces the measured values.

Now the per-head numbers mean something. Still frozen, still no optimiser: this only measures.

What it answers, per head, over real text:
  * gather norm        - is the row live, and how strong
  * gate value         - what the model multiplies the write by
If head 3 is quiet in BOTH, it is a weight/coverage fact. If its gather is quiet but its gate is open,
the suppression is downstream and a targeted edit could reach it.
"""
import json

import modal

app = modal.App("whittle-engram-gate2")

HF_CACHE = modal.Volume.from_name("whittle-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformers==5.22.1", "accelerate", "safetensors",
                 "huggingface_hub", "numpy", "tokenizers", "datasets")
    .env({"HF_HOME": "/cache"})
    .add_local_file("engram_load.py", "/root/engram_load.py")
)

REPO = "whittleOrg/Whittle-Qwen-3.8-45B-A3B"


@app.function(image=image, gpu="H200", volumes={"/cache": HF_CACHE}, timeout=3600)
def probe(n_tokens: int = 1024):
    import sys

    import numpy as np
    import torch
    from datasets import load_dataset

    sys.path.insert(0, "/root")
    from engram_load import load_whittle

    model, tok, info = load_whittle(verify=True)
    print("load info:", json.dumps({k: v for k, v in info.items() if k != "modules_fixed"}))

    ds = load_dataset("tatsu-lab/alpaca", split="train[:300]")
    text = "".join("User: %s\nAssistant: %s\n" % (r["instruction"], r["output"]) for r in ds)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=n_tokens).input_ids.to("cuda")
    print("tokens:", tuple(ids.shape))

    ple = None
    for name, m in model.named_modules():
        if type(m).__name__ == "Qwen4ExpTextPLELayer":
            ple = m
    emb_mod = ple.ple_embedding

    cap = {}
    hs = [
        emb_mod.ngram_embedding.register_forward_pre_hook(
            lambda m, a: cap.__setitem__("ids", a[0].detach())),
        emb_mod.register_forward_hook(
            lambda m, a, o: cap.__setitem__("gather", o.detach())),
        ple.norm_key.register_forward_hook(
            lambda m, a, o: cap.__setitem__("key", o.detach())),
        ple.norm_query.register_forward_hook(
            lambda m, a, o: cap.__setitem__("query", o.detach())),
        ple.value_proj.register_forward_hook(
            lambda m, a, o: cap.__setitem__("value", o.detach())),
    ]
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    for h in hs:
        h.remove()

    W = emb_mod.ngram_embedding.weight
    rows = cap["ids"][0].cpu().numpy().astype(np.int64)

    print("\n  %-5s %-8s %10s %10s %10s %8s %8s"
          % ("head", "kind", "gather", "value", "gate", "empty", "chance"))
    per_head = []
    for h in range(8):
        rid = torch.as_tensor(rows[:, h], device=W.device)
        g = W[rid].float()
        nrm = g.detach().norm(dim=1).cpu().numpy()
        st = {
            "head": h,
            "kind": "bigram" if h < 4 else "trigram",
            "gather_mean": float(nrm.mean()),
            "gather_median": float(np.median(nrm)),
            "empty_frac": float((nrm == 0).mean()),
        }
        per_head.append(st)
        print("  %-5d %-8s %10.4f %10s %10s %7.1f%% %8s"
              % (h, st["kind"], st["gather_mean"], "-", "-", 100 * st["empty_frac"], "-"))

    # per-head share of the gathered 2048-dim vector's energy
    gath = cap["gather"].float().reshape(-1, 8, 256)
    energy = gath.pow(2).sum(-1).mean(0).cpu().numpy()
    share = 100 * energy / energy.sum()
    print("\nper-head share of gather energy: %s" % [round(float(v), 2) for v in share])
    for h in range(8):
        per_head[h]["energy_share_pct"] = float(share[h])

    # the gate, computed exactly as the model does
    kv = cap["key"].float()
    qv = cap["query"].float()
    hs_, hc = ple.hidden_size, ple.hc_count
    kn = kv.reshape(-1, hc, hs_)
    qn = qv.reshape(-1, hc, hs_)
    gate_raw = (kn * qn).sum(-1) / (hs_ ** 0.5)
    gate = torch.sigmoid(gate_raw.abs().clamp_min(1e-6).sqrt() * gate_raw.sign())
    print("\ngate: mean=%.4f std=%.4f min=%.4f max=%.4f  >0.5 on %.1f%%  <0.1 on %.1f%%"
          % (float(gate.mean()), float(gate.std()), float(gate.min()), float(gate.max()),
             100 * float((gate > 0.5).float().mean()), 100 * float((gate < 0.1).float().mean())))
    print("gate per hc copy:", [round(float(v), 4) for v in gate.mean(0)])

    # value: is the write itself quiet?
    val = cap["value"].float()
    print("value_proj out: mean|v|=%.4f  median|v|=%.4f"
          % (float(val.norm(dim=-1).mean()), float(val.norm(dim=-1).median())))

    # the gated write magnitude, per token
    write = gate.unsqueeze(-1) * val.unsqueeze(-2)
    print("gated write |.|^2 mean = %.4f" % float(write.pow(2).mean()))
    print("value |.|^2 mean       = %.4f  (gate is multiplying it down by %.2fx)"
          % (float(val.pow(2).mean()), float(write.pow(2).mean() / max(1e-9, float(val.pow(2).mean())))))

    loss = float(torch.nn.functional.cross_entropy(
        out.logits[:, :-1].reshape(-1, out.logits.shape[-1]).float(), ids[:, 1:].reshape(-1)))
    print("\nloss on this sample: %.4f" % loss)

    return {"tokens": int(ids.shape[1]), "info": {k: v for k, v in info.items() if k != "modules_fixed"},
            "per_head": per_head, "loss": loss,
            "gate_summary": {"mean": float(gate.mean()), "std": float(gate.std()),
                             "frac_gt_0.5": float((gate > 0.5).float().mean()),
                             "frac_lt_0.1": float((gate < 0.1).float().mean())},
            "value_norm_mean": float(val.norm(dim=-1).mean())}


@app.local_entrypoint()
def main(n_tokens: int = 1024):
    print(json.dumps(probe.remote(n_tokens=n_tokens), indent=1))
