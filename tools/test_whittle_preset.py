"""Verify the Whittle preset resolves correctly, by executing the real definitions out of setup.py.

setup.py imports the strata_control package at module level, which is not present in a bare
checkout, so this pulls just the data tables and the few helper functions out of the AST and
runs them. That keeps the check honest: the numbers below come from setup.py's own objects.
"""
import ast
import os

SRC = "/opt/data/work/strata-fork/setup.py"
WANT = {"FAMILIES", "MODELS", "WHITTLE_FILES", "WHITTLE_ENGINE", "HF_REVISIONS", "RECOMMENDED_SIZES"}
FUNCS = {"model_file", "model_shards", "recommended_size", "size_tag", "mem_have", "hf"}

src = open(SRC, encoding="utf-8").read()
tree = ast.parse(src)

# The tables reference a web of module-level constants (engine tuples, repo names, file dicts).  Rather
# than chase each one, take every simple assignment and evaluate them in dependency order: retry the
# ones that raise NameError until a pass makes no progress.  Anything still unresolved is a network or
# control-plane object we do not want, and is left out.
FUNCS = {"model_file", "model_shards", "recommended_size", "size_tag", "mem_have", "hf"}

src = open(SRC, encoding="utf-8").read()
tree = ast.parse(src)

prelude = '''
HF_DEFAULT = "https://huggingface.co"
import os
def hf_endpoint():
    return (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or HF_DEFAULT
def tier_pick(ram, vram=0.0):
    return None
'''

ns = {}
exec(compile(prelude, "prelude", "exec"), ns)

func_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS]
func_nodes.sort(key=lambda n: 0 if n.name == "hf" else 1)

assign_nodes = [n for n in tree.body if isinstance(n, ast.Assign)
                and all(isinstance(t, ast.Name) for t in n.targets)]

pending = func_nodes + assign_nodes
while pending:
    progress, still = False, []
    for node in pending:
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "setup_slice", "exec"), ns)
            progress = True
        except (NameError, AttributeError, KeyError, TypeError, SyntaxError):
            still.append(node)
    if not progress:
        break
    pending = still

missing = sorted({t.id for n in pending if isinstance(n, ast.Assign) for t in n.targets})

names = ["Whittle-Q8_0", "Whittle-Q5_K_M", "Whittle-Q4_K_M", "Whittle-Q3_K_M"]
f = ns["FAMILIES"]["whittle"]

print("family :", f["title"], "|", f["by"])
print("hf     :", f["hf"])
print("file   :", f["file"], "| shards:", f["shards"])
print("tag    :", f["tag"], "| pack_args:", f["pack_args"], "| sha256:", len(f["sha256"]))
print("license:", f["license"][:72], "...")
print()
for n in names:
    print("  %-16s %-40s sh=%d runnable=%s" % (
        n, ns["model_file"](f, n, 1), ns["model_shards"](f, n), ns["MODELS"][n].get("runnable", True)))
print()
print("URL:", f["hf"] + ns["model_file"](f, "Whittle-Q8_0", 1))
for ram in (64, 32, 16):
    print("recommend %2d GB RAM -> %s" % (ram, ns["recommended_size"]("whittle", names, float(ram), 0.0)))
for n in names:
    print("  tag %-16s%s" % (n, ns["size_tag"]("whittle", n)))
print()
print("revision pinned:", ns["HF_REVISIONS"]["logic65/Whittle-Qwen-3.8-45B-A3B-GGUF"])
print("engine req     :", ns["WHITTLE_ENGINE"])
print()
print("arena check (stated GB -> GiB vs engine's measured 31.88 GiB for Q8_0):")
for n in names:
    a = ns["MODELS"][n]["arena_gb"]
    print("  %-16s %6.2f GB = %5.2f GiB" % (n, a, a / 1.073741824))
