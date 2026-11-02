"""Dump every qwen4exp.* key in Whittle's GGUF, verbatim, plus the tensor table.

The published transformers reference builds head_vocab_sizes as DISTINCT primes
(_find_nth_prime_after(base-1, head+1)), while the GGUF's arrays look like eight identical 4,880,000s.
Those cannot both be right, and the whole plan depends on which one the trained weights were built with:
if the sizes really are equal, all four heads of an order share one hash address; if they are distinct
primes, they do not, and the "shared address" finding is an artefact of my own reader.

No interpretation here - just the bytes, so the claim can be checked rather than believed.
"""
import struct
import sys

FIX = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
ARR = {0: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
FMT = {0: "<b", 1: "<B", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def main(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver, = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def rs():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def val(t):
            if t == 8:
                return rs()
            if t == 9:
                et, = struct.unpack("<I", f.read(4))
                cnt, = struct.unpack("<Q", f.read(8))
                if et in ARR:
                    import array
                    a = array.array(ARR[et])
                    a.frombytes(f.read(cnt * FIX[et]))
                    return list(a)
                return [val(et) for _ in range(cnt)]
            return struct.unpack(FMT[t], f.read(FIX[t]))[0]

        meta = {}
        for _ in range(n_kv):
            k = rs()
            t, = struct.unpack("<I", f.read(4))
            meta[k] = val(t)

        print("GGUF v%d  tensors=%d  kv=%d" % (ver, n_tensors, n_kv))
        print("\n--- every qwen4exp/ple/ngram key, verbatim ---")
        for k in sorted(meta):
            if any(s in k.lower() for s in ("qwen4exp", "ple", "ngram", "expert", "head")):
                v = meta[k]
                if isinstance(v, list) and len(v) > 12:
                    print("  %-52s list[%d] %s ... %s" % (k, len(v), v[:10], v[-3:]))
                else:
                    print("  %-52s %s" % (k, v))

        print("\n--- tensor table ---")
        for _ in range(n_tensors):
            name = rs()
            nd, = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            tt, = struct.unpack("<I", f.read(4))
            off, = struct.unpack("<Q", f.read(8))
            if "ple" in name or "ngram" in name:
                print("  %-40s dims=%s type=%d off=%d" % (name, dims, tt, off))

        # the decisive arithmetic: does the row count equal 8 equal heads, or 8 distinct primes?
        key = next((k for k in meta if "head_vocab" in k.lower() or "vocab_sizes" in k.lower()), None)
        if key:
            sizes = meta[key]
            print("\n--- decisive: %s ---" % key)
            print("  values      :", sizes)
            print("  distinct?   :", len(set(sizes)) == len(sizes))
            print("  sum         :", sum(sizes))
            print("  == 8*4880000:", sum(sizes) == 8 * 4880000)
            print("  equal sizes :", len(set(sizes)) == 1)
        else:
            print("\n  NO head_vocab_sizes key found - dumping all keys containing 'vocab':")
            for k in sorted(meta):
                if "vocab" in k.lower():
                    print("   ", k, "=", meta[k])


if __name__ == "__main__":
    main(sys.argv[1])
