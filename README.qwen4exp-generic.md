# strata-qwen4exp-generic

A fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) that makes the compiled
model shape a **build choice** instead of a hardcoded constant, so more than one `qwen4exp`
model can run on the same engine.

Upstream is an excellent MoE inference engine — expert cache in VRAM, all experts pinned in
RAM, the n-gram (PLE) table streamed from SSD, MTP speculative decoding — and it carries a
complete SYCL backend for Intel Arc. But it compiles its kernels for exactly one shape:
**Qwen3.8-Flash-Next** (`n_embd` 2560, `n_ff` 640, 48 layers, 512 experts, 48 ssm value
heads). Every kernel that pins a dimension `static_assert`s it against
`core::kCompiledGeometry`, and `read_geometry` refuses any file whose metadata disagrees.

That means models that are *architecturally identical* — same hyper-connections, same
indexer, same GDN/SSM mixer, same MoE trunk — cannot run if their widths differ, even
though nothing in the algorithm depends on those widths.

This fork does not change the algorithm. It makes the shape configurable.

## What changed

| commit | change |
|---|---|
| `ddfd85f` | `include/strata/core/geometry_build.hpp` + `STRATA_GEOM_PROFILE` in `sycl/CMakeLists.txt`; `geometry.hpp` defaults now derive from the profile |
| `d843950` | `include/strata/kernels/cpu/expert.hpp` — `H`/`FF`/`NE` from the profile (everything downstream already derived from them) |
| `25db16e` | the remaining pinned literals in the engine build path: `s2_expert_grouped`, `fused_gr`, both `prefill` files, `native_ple_postops`, `ple`, `iq_kernels`, `expert_source` |

The pattern throughout: a literal that used to be `2560` becomes `STRATA_N_EMBD`, and a
size that used to be `1280 * 2560` becomes `2 * STRATA_N_FF * STRATA_N_EMBD`. On the
default profile these are bit-identical to upstream — the flashnext build is unchanged.

## Profiles

```
-DSTRATA_GEOM_PROFILE=flashnext   (default)  n_embd 2560  48 layers  n_ff 640  512 experts  48 ssm_v_heads
-DSTRATA_GEOM_PROFILE=w45                    n_embd 2048  40 layers  n_ff 512  256 experts  32 ssm_v_heads
```

`w45` targets [`logic65/Whittle-Qwen-3.8-45B-A3B`](https://huggingface.co/logic65/Whittle-Qwen-3.8-45B-A3B)
— a Qwen3.6-35B-A3B base with 76 experts restored into every layer and a ~10 B-parameter
n-gram table. Its GGUF reports `general.architecture = qwen4exp`, the same family as
Flash-Next, which is why it is a candidate for this engine at all.

Any single axis can be overridden without adding a profile:

```sh
cmake -S sycl -B build -DSTRATA_N_EMBD=3072 -DSTRATA_N_FF=768 -DSTRATA_N_HEAD=24
```

Axes that are already identical on both models (and therefore never needed changing):
`hc`=4, `hc_lr`=320, `head_dim`=256, `n_rot`=64, `idx_q_heads`=4, `idx_key_dim`=128,
`idx_block`=4, `ssm_state_size`=128, `ssm_k_heads`=16, `ssm_d_conv`=4, `qsa_interval`=4,
`n_head_kv`=2. They are still exposed as macros so an unrelated model can vary them.

Derived values the kernels share:

```c
STRATA_HC_DIM             (hc * n_embd)                      10240 ->    8192
STRATA_SSM_CONV_CHANNELS  2*state*k_heads + value_dim         10240 ->    8192
STRATA_EXPERT_BLOB        3 * n_embd * n_ff * 18 / 64       1382400 ->  884736
STRATA_S26_XMAX           n_embd / 32                            80 ->      64
```

## Safety: the check is not weakened

A file whose metadata disagrees with the compiled profile is **still refused at load** by
`geometry_unsupported`, naming the keys that differ. This fork widens which shapes the build
can be *pointed at*; it does not let a mismatched file run. If you build `flashnext` and
load a `w45` model you get the same clear error as upstream.

## Build (Intel Arc)

```sh
# oneAPI 2026.1's MKL ships without include/ -- point MKLROOT at a complete install
export MKLROOT=/opt/intel/oneapi/mkl/2026.0

cmake -S sycl -B build-w45 -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DSTRATA_SYCL_PARITY=OFF \
      -DSTRATA_SYCL_AOT=bmg-g21 \
      -DSTRATA_GEOM_PROFILE=w45
cmake --build build-w45 --target strata -j "$(nproc)"
```

`bmg-g21` is B580 / B570 / **Arc Pro B60**; `bmg-g31` is the Arc Pro B70. Omit
`-DSTRATA_SYCL_AOT` to JIT from SPIR-V (needed for Alchemist A-series, which has no FP64).

Verified: builds on an Arc Pro B60 (PCI `8086:e211`, `xe`, oneAPI 2026.1.1).

## Status and what is still missing

The engine **compiles** for the `w45` profile, and a full Whittle pack builds end to end.
What remains is a *quantisation-format* limit, not a geometry one.

### The pack builds

`tools/iq_pack.py` is the geometry-agnostic packer (it derives `n_layers` and `n_expert`
from the file's own routers, and keeps quantized tensors in GGUF form). On a Q3_K_M Whittle:

```
index.txt: 901 tensors, 252 served natively, arena 0.91 GiB
conversions.json: 404 tensors converted (60 exact, 344 rounded; max |err| 0.000964)
compat-bf16: 364 tensors, 0.91 GiB; expert and PLE table bytes unchanged
RC=0  ->  dense.bin, index.txt, native_experts.txt, tokenizer/, conversions.json
```

Two prerequisites that cost a cycle each:

- `STRATA_GGUF_PY` must point at a llama.cpp `gguf-py` (the tool does not vendor it).
- `strata_tokenizer.py` imports `regex`, which is not a stock-Python module; without it
  `iq_pack.py` dies *after* writing `dense.bin` but *before* `native_experts.txt`, and the
  pack then reads as a canonical Q2_0 pack — which is what makes the AVX-512 gate fire on
  a machine that would otherwise have been fine.

### The remaining blocker: a GPU page fault inside prefill

With the Q8_0 build the engine now gets all the way to a live session and then faults
inside the prompt path:

```
strata generate: GPU 0: Intel(R) Arc(TM) Pro B60 Graphics, compute capability 20.1
strata generate: PLE on, table 39040000 rows
strata generate: session is up (engine 0.1.53)
level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
  at sycl/src/prefill/prefill.cpp:5051
```

`dmesg` shows the cause is a real out-of-bounds access, not a driver flake:

```
xe 0000:03:00.0: [drm] Tile0: GT0: Fault response: Unsuccessful -EINVAL
xe 0000:03:00.0: [drm] exec queue reset detected
```

Six classes of Flash-Next literal have been found and fixed in the prompt path
(`prefill.cpp` buffer plan, `kernels.dp.cpp` blob layout + QSA head indexing,
`moe_fused.dp.cpp` / `moe_fused_iq.dp.cpp` accumulator row stride, `gdn_xe.hpp`
v_heads). **The fault survives all of them**, so at least one more remains — the most
likely place is a stride that is not a plain literal (a derived `ld`, a tile count, or a
`static_assert`-free constant in an ESIMD/XMX kernel).

Not established: whether this fault is *ours* or environmental. No model has yet run
end-to-end on this engine on this box, so there is no known-good control. Getting one —
a Flash-Next Q2_0 pack, which is what the SYCL port actually targets — would separate
"my port has another bug" from "this build does not run on this machine at all". Until
that control exists, treat the port as unproven past session setup.

### The expert-format limit (independent of the above)

`native_expert_supported()` requires the gate/up and down formats to be in
`STRATA_GU_FMTS` / `STRATA_D_FMTS`. Read from the GGUF headers, Whittle's quants use:

| quant | gate/up | down | GPU kernels? |
|---|---|---|---|
| Q3_K_M | Q3_K | Q4_K ×38, Q5_K ×2 | **no** — Q3_K not in GU_FMTS, Q5_K not in D_FMTS |
| Q4_K_M | Q4_K | Q4_K ×20, Q6_K ×20 | **no** — Q6_K not in D_FMTS |
| Q5_K_M | Q5_K | Q5_K ×20, Q6_K ×20 | **no** — Q5_K/Q6_K not in D_FMTS |
| **Q8_0** | **Q8_0** | **Q8_0** | **yes** — type 8 is in both lists |

So the engine reaches Whittle only from the **Q8_0** build, and only on a box with real
RAM: the expert arena is 31.88 GiB, which does not fit the 31 GB test box alongside the
dense set. `--stream-experts` (no host arena, experts pulled from the GGUF on demand)
is the workable mode there.

Adding the K-quant expert formats is real kernel work (`Fmt<T>` specialisations plus the
grouped dot), not a table edit — and it would be premature until the prefill fault above
is understood.

Not tested: numerics against a reference, throughput, or the Qwen3.6 (`qwen35moe`) path.
This is a geometry-enablement fork, not a validated model port.

## Upstreaming

The geometry profile is deliberately additive and profile-agnostic — `flashnext` reproduces
upstream bit-for-bit. The pieces that would be worth offering upstream are
`geometry_build.hpp` and the `STRATA_GEOM_PROFILE` CMake option; the specific `w45` numbers
are just one instantiation. A parameterised expert arena in `pack_layer.py` would help
every non-Flash-Next `qwen4exp` model, not only this one.

## License

MIT, as upstream. Original work by Niko1221 and contributors.
