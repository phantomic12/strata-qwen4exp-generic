"""How many tokens does it take to put real mass into a 4.88M-bucket head?

The retrain's cost is driven by DATA, not by FLOPs: a bucket only stops being empty when a real
bigram lands in it. So the decisive number is how fast distinct bigrams accumulate with corpus size.

Duplicates matter. A 44k-token sample gives 0.571 distinct bigrams/token, but that ratio falls as the
corpus grows (Zipfian), so extrapolating from it overestimates coverage badly. This measures the
actual curve on a larger corpus and finds the exponent.

Reports distinct bigrams at 100k/1M/5M tokens, fits distinct ~ n^alpha, and projects the tokens needed
to fill a 4.88M-bucket head to 50% and 90% (expected distinct = -ln(1-frac) * buckets, by the
coupon-collector bound for uniform hashing).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = "whittleOrg/Whittle-Qwen-3.8-45B-A3B"
BUCKETS = 4_880_000
OUT = Path("data")


def main():
    tk = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    target = 5_000_000
    chunks, total = [], 0
    for r in ds:
        t = r["text"]
        if len(t) < 200:
            continue
        chunks.append(t)
        total += len(t)
        if total > target * 4:      # chars ~ 4x tokens
            break
    print("text: %d docs, %d chars" % (len(chunks), total))

    ids = tk.encode("".join(chunks)).ids
    print("tokens: %d" % len(ids))

    eos = 248044
    arr = np.array(ids, dtype=np.int64)

    pts = []
    for n in (100_000, 1_000_000, 5_000_000):
        if n > len(arr):
            continue
        sub = arr[:n]
        # bigrams, skipping any pair that spans an eos
        a, b = sub[:-1], sub[1:]
        ok = (a != eos) & (b != eos)
        pairs = a[ok] * 248070 + b[ok]
        d = int(np.unique(pairs).size)
        pts.append((n, d))
        print("  %9d tokens -> %10d distinct bigrams  (%.3f/token)" % (n, d, d / n))

    if len(pts) >= 2:
        xs = np.log([p[0] for p in pts])
        ys = np.log([p[1] for p in pts])
        alpha, icpt = np.polyfit(xs, ys, 1)
        print("\nfit: distinct ~ n^%.3f  (constant %.3g)" % (alpha, math.exp(icpt)))

        print("\ntokens needed to fill a %d-bucket head:" % BUCKETS)
        for frac in (0.5, 0.9):
            need = -math.log(1 - frac) * BUCKETS
            # invert distinct = C * n^alpha
            n_req = math.exp((math.log(need) - icpt) / alpha)
            print("  %3.0f%% fill -> %.2fM distinct bigrams -> %.1fM tokens (1 pass)"
                  % (frac * 100, need / 1e6, n_req / 1e6))
            for reps in (3, 10):
                print("       x%d passes over shuffled data -> %.0fM tokens seen"
                      % (reps, n_req * reps / 1e6))

    json.dump({"tokens": int(len(arr)), "points": pts,
               "buckets": BUCKETS}, open(OUT / "distinct_bigrams.json", "w"), indent=1)


if __name__ == "__main__":
    main()
