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

### The remaining blocker: expert tensor types

`native_expert_supported()` requires the gate/up and down formats to be in
`STRATA_GU_FMTS` / `STRATA_D_FMTS`. Read from the GGUF headers, Whittle's quants use:

| quant | gate/up | down | GPU kernels? |
|---|---|---|---|
| Q3_K_M | Q3_K | Q4_K ×38, Q5_K ×2 | **no** — Q3_K not in GU_FMTS, Q5_K not in D_FMTS |
| Q4_K_M | Q4_K | Q4_K ×20, Q6_K ×20 | **no** — Q6_K not in D_FMTS |
| Q5_K_M | Q5_K | Q5_K ×20, Q6_K ×20 | **no** — Q5_K/Q6_K not in D_FMTS |
| **Q8_0** | **Q8_0** | **Q8_0** | **yes** — type 8 is in both lists |

So the engine reaches Whittle only from the **Q8_0** build. The K-quant expert types are in
`is_iq()` but absent from the two format tables, i.e. the grouped kernels have no
instantiations for them.

Two ways forward, in order of cost:

1. **Use Q8_0** (48 GB). Works with no source change. Expert arena ~9.1 GiB (256 experts x
   3 roles x 2048 x 512 at 8.5 bpw), so it needs a box with real RAM — the i7-10700's 31 GB
   is marginal for the arena plus the host-side dense set, and the PLE table streams from disk.
2. **Add the K-quant expert formats** to the two tables and write the kernels. Q4_K/Q6_K
   would then cover Q4_K_M, and Q3_K/Q5_K cover Q3_K_M. This is real kernel work
   (`Fmt<T>` specialisations + the grouped dot), not a table edit.

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
