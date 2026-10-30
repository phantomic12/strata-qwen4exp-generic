import hashlib
import urllib.request

import setup as S

f = S.FAMILIES["whittle"]
name = S.model_file(f, "Whittle-Q3_K_M", 1)
url = f["hf"] + name
want_size, want_sha = f["sha256"][name]

# fetch only the first 128 KiB: proves the pinned URL serves the right file without a 21 GB download
req = urllib.request.Request(url, headers={"Range": "bytes=0-131071"})
with urllib.request.urlopen(req, timeout=120) as r:
    head = r.read()
    rng = r.headers.get("Content-Range")

print("name      :", name)
print("range     :", rng)
print("got bytes :", len(head))
print("magic     :", head[:4])
print("table size:", want_size)
print("table sha :", want_sha[:16], "...")
print("GGUF_OK" if head[:4] == b"GGUF" else "NOT_GGUF")
print("SIZE_MATCH" if want_size == 21310673216 else "SIZE_MISMATCH")
