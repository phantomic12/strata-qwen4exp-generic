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

The engine **compiles** for the `w45` profile. Two things remain before a Whittle
end-to-end run, both outside the kernel geometry:

1. **The pack builder is Flash-Next-shaped.** `tools/pack_layer.py` asserts
   `t.type_name == "Q2_0"`, plans 48 layers regardless of the file, and reads a module-level
   `SHARD1` constant instead of the `--gguf` argument. Its expert arena is the fixed
   1,382,400-byte layout (512 experts x 640). `STRATA_EXPERT_BLOB` now gives the right
   number for w45 (884,736) but the packer does not consume it yet. For a Q3_K/Q4_K model
   the native pack path (`native_experts.txt`) is the intended route.

2. **`cpu_require_expert_support()` demands AVX-512** (VNNI + VBMI). The test box is an
   i7-10700 (Comet Lake, AVX2 only), so it exits. The gate only fires for the canonical
   Q2_0 pack — a legitimate native pack skips it — which is why fix (1) unblocks (2) on
   AVX2 hardware.

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
