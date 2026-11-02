"""Load the table straight from the shards instead, sidestepping from_pretrained entirely.

from_pretrained cannot help us here: the module sizes the embedding from config (39,040,640 rows) while
the checkpoint is 39,040,000 rows, so the tensor never gets assigned. `engram_fixload.py` tries to patch
after the fact, which is fragile - if the resize happens after loading, the weights are simply gone.

The reliable route is the one the author's own `whittle_load.py` takes: read the shards yourself and
place the rows. This does exactly that, and then VERIFIES the placement by checking that the known-empty
rows of head 3 really are empty - a freshly initialised table has no such runs.
"""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "whittleOrg/Whittle-Qwen-3.8-45B-A3B"
SHARD_KEY = "model.ngram_embedding.shard_%d.weight"

TRUE_VOCAB = [4_880_000] * 8
TRUE_OFFSETS = [4_880_000 * i for i in range(8)]
TRUE_MULT = [32473892291597, 22010589867355, 6403636407469]


def load_whittle(device_map="auto", dtype=torch.bfloat16, verify=True):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(REPO, allow_patterns=["*.safetensors", "*.json"])
    model = AutoModelForCausalLM.from_pretrained(
        REPO, dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    tok = AutoTokenizer.from_pretrained(REPO)

    # locate the PLE module
    ple_name, ple = None, None
    for name, mod in model.named_modules():
        if type(mod).__name__ == "Qwen4ExpTextPLELayer":
            ple_name, ple = name, mod
    if ple is None:
        raise RuntimeError("no Qwen4ExpTextPLELayer found")

    emb = ple.ple_embedding
    want = sum(TRUE_VOCAB)

    # 1. gather the real table from the shards, on CPU, bf16
    import glob
    files = sorted(glob.glob(path + "/*.safetensors"))
    parts = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as fh:
            for key in fh.keys():
                if key.startswith("model.ngram_embedding.shard_"):
                    parts[key] = fh.get_tensor(key)
    n_shards = len(parts)
    table = torch.cat([parts[SHARD_KEY % i] for i in range(n_shards)], dim=0)
    print("[fixload] read %d shards -> %s %s" % (n_shards, tuple(table.shape), table.dtype))
    assert table.shape[0] == want, "checkpoint table is %d rows, expected %d" % (table.shape[0], want)

    # 2. put the layout buffers back to the trained values, then the rows
    emb.head_vocab_sizes = list(TRUE_VOCAB)
    emb.head_offsets = list(TRUE_OFFSETS)

    def _devof(obj, default):
        return obj.device if isinstance(obj, torch.Tensor) else default

    def _set(name, values):
        """Replace a buffer keeping its original device/dtype - a CPU tensor here breaks the forward."""
        old_v = getattr(emb, name, None)
        dev = _devof(old_v, "cpu")
        dt = old_v.dtype if isinstance(old_v, torch.Tensor) else torch.long
        if isinstance(old_v, (list, tuple)):
            setattr(emb, name, list(values))
        else:
            setattr(emb, name, torch.tensor(values, dtype=dt, device=dev))

    _set("layer_multipliers", TRUE_MULT)
    _set("ngram_heads_vocab_sizes", TRUE_VOCAB)
    _set("ngram_heads_offsets", TRUE_OFFSETS)

    old = emb.ngram_embedding
    target_device = old.weight.device
    emb.ngram_embedding = torch.nn.Embedding(want, old.weight.shape[1], dtype=dtype).to(target_device)
    with torch.no_grad():
        emb.ngram_embedding.weight.copy_(table.to(dtype=dtype, device=target_device))
    del table, parts

    info = {"module": ple_name, "embedding_rows": int(emb.ngram_embedding.weight.shape[0])}

    if verify:
        W = emb.ngram_embedding.weight
        # head 3 is 79.5% empty and head 7 is 65.3% empty in the real table; a random init is 0% empty.
        # If these numbers come out anywhere near the measured values, the placement is correct.
        per_head = []
        for h in range(8):
            lo, n = TRUE_OFFSETS[h], TRUE_VOCAB[h]
            idx = torch.arange(lo, lo + n, 9973, device=W.device)
            nrm = W[idx].float().norm(dim=1)
            per_head.append(round(100 * float((nrm > 0).float().mean()), 1))
        info["live_pct_per_head"] = per_head
        expect = [93.4, 93.4, 93.4, 20.5, 99.9, 99.9, 99.9, 34.7]
        ok = all(abs(a - b) < 6 for a, b in zip(per_head, expect))
        info["placement_verified"] = ok
        print("[fixload] live%% per head: %s" % per_head)
        print("[fixload] expected      : %s" % expect)
        print("[fixload] PLACEMENT VERIFIED: %s" % ok)
        if not ok:
            raise RuntimeError("table placement did not reproduce the measured occupancy")

    return model, tok, info


if __name__ == "__main__":
    m, t, i = load_whittle()
    print(json.dumps(i, indent=1))
