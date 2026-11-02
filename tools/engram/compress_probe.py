"""How much would V4.1's compressed-token trick shrink Whittle's n-gram space?

V4.1-Flash-1017 hashes n-grams over a COMPRESSED id space (~99k for a 129k vocab): every token id is
normalized (NFKC, strip accents, lowercase, collapse whitespace) and mapped to a canonical id, so
" The", "the" and "THE" hash identically. That collapses the bucket space and puts variants where the
table's mass already is.

Whittle hashes over the RAW 248,070-token vocab. Its bigram space is 248070^2 = 6.15e10 against
4,880,000 buckets per head - a 12,600:1 overload, so buckets collide by construction. Compressing the
vocab shrinks that overload.

This measures, on Whittle's real tokenizer:
  * the compressed vocab size under V4.1's exact normalizer chain
  * the bigram-space overload before and after
  * how often a real corpus's bigrams merge after compression (collision/merge rate)
No GPU, no table read.
"""
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Regex, Tokenizer, normalizers

REPO = "whittleOrg/Whittle-Qwen-3.8-45B-A3B"
BUCKETS_PER_HEAD = 4_880_000


def build_compressed_map(tk):
    """V4.1's normalizer chain, applied per token id, with the space sentinel it uses."""
    sentinel = "\ue000"
    norm = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    backend = getattr(tk, "backend_tokenizer", tk)   # raw tokenizers.Tokenizer IS the backend
    n = tk.get_vocab_size()
    key_to_new, lookup = {}, np.zeros(n, dtype=np.int64)
    for tid in range(n):
        text = backend.decode([tid], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(tid)
        else:
            n = norm.normalize_str(text)
            key = n if n else text
        if key not in key_to_new:
            key_to_new[key] = len(key_to_new)
        lookup[tid] = key_to_new[key]
    return lookup, len(key_to_new)


def main():
    tk = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
    vocab = tk.get_vocab_size()
    print("raw vocab: %d" % vocab)

    lookup, ncomp = build_compressed_map(tk)
    print("compressed vocab: %d  (%.1f%% of raw, %.1fx smaller)"
          % (ncomp, 100 * ncomp / vocab, vocab / ncomp))

    print("\nbigram bucket space vs %d buckets/head:" % BUCKETS_PER_HEAD)
    for label, n in (("raw", vocab), ("compressed", ncomp)):
        space = n * n
        print("  %-11s %12d bigrams  -> %.0f:1 overload  (%.4f%% of space addressable)"
              % (label, space, space / BUCKETS_PER_HEAD, 100 * BUCKETS_PER_HEAD / space))

    print("\ntrigram space:")
    for label, n in (("raw", vocab), ("compressed", ncomp)):
        print("  %-11s %.3e" % (label, float(n) ** 3))

    # real-corpus merge rate: do distinct token bigrams collapse onto fewer compressed bigrams?
    for corpus in ("tokens_wiki.npy", "tokens_chat.npy"):
        p = Path("data") / corpus
        if not p.exists():
            continue
        tok = np.load(p).astype(np.int64)
        comp = lookup[tok]
        raw_pairs, comp_pairs = set(), set()
        for i in range(1, len(tok)):
            raw_pairs.add((int(tok[i - 1]), int(tok[i])))
            comp_pairs.add((int(comp[i - 1]), int(comp[i])))
        print("\n%s: %d raw bigrams -> %d compressed (%.1f%% merge)"
              % (corpus, len(raw_pairs), len(comp_pairs),
                 100 * (1 - len(comp_pairs) / len(raw_pairs))))

    np.save("data/whittle_token_map.npy", lookup)


if __name__ == "__main__":
    main()
