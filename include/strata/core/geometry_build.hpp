// include/strata/core/geometry_build.hpp - build-time geometry selection for non-Flash-Next qwen4exp models.
//
// Upstream (Niko1221/Strata) compiles the kernels for ONE shape: Qwen3.8-Flash-Next's
// (n_embd 2560, n_ff 640, 48 layers, 512 experts, 48 ssm value heads).  Every kernel that pins a
// dimension `static_assert`s it against `core::kCompiledGeometry`, and `read_geometry` refuses a file
// whose metadata disagrees -- so a different qwen4exp model cannot run, even though the architecture
// (hyper-connections, the indexer, the GDN/SSM mixer, the MoE trunk) is identical.
//
// This header makes that shape a build choice instead of a constant.  Two profiles exist:
//
//   STRATA_GEOM_PROFILE=flashnext  (default) - upstream's shape, unchanged on every axis
//   STRATA_GEOM_PROFILE=w45                  - Whittle-Qwen-3.8-45B-A3B (logic65/Whittle)
//
// and any single axis can be overridden from the command line, e.g.
//   -DSTRATA_N_EMBD=3072 -DSTRATA_N_FF=768
//
// Everything else in the tree derives from these macros, so one tree builds either model and the
// static_asserts in the kernels keep working -- they now check against the selected shape rather than
// against a hardcoded literal.
//
// WHY MACROS AND NOT A RUNTIME STRUCT: the kernels need these numbers as compile-time constants (tile
// sizes, register blocking, `template <int N>`).  The engine already reads the file's real geometry into
// `core::ModelGeometry` at load; this header exists so that the *compiled* shape can match it.  A model
// whose metadata disagrees with the compiled profile is still refused, loudly, by
// `geometry_unsupported` -- we widen which shapes the build can be pointed at, we do not weaken the check.
#pragma once

// ---------------------------------------------------------------------------------------------------
// Profile selection
// ---------------------------------------------------------------------------------------------------
#if !defined(STRATA_GEOM_PROFILE)
#define STRATA_GEOM_PROFILE flashnext
#endif

#define STRATA_GEOM_FLASHNEXT 1
#define STRATA_GEOM_W45 2

// Resolve the profile token to a number without requiring the caller to know the numbering.
#if STRATA_GEOM_PROFILE_PICK == STRATA_GEOM_FLASHNEXT  // never true; keeps the tokens usable below
#endif

// ---------------------------------------------------------------------------------------------------
// Defaults per profile.  Each value is "#ifndef"-guarded so -D overrides a single axis.
//
//   field                 flashnext   w45     source
//   n_embd                    2560    2048   qwen4exp.embedding_length
//   n_layers                    48      40   qwen4exp.block_count
//   n_ff (expert)              640     512   qwen4exp.expert_feed_forward_length
//   n_expert                   512     256   qwen4exp.expert_count
//   n_expert_used               10       8   qwen4exp.expert_used_count
//   n_head                      24      16   qwen4exp.attention.head_count
//   n_head_kv                    2       2   qwen4exp.attention.head_count_kv
//   ssm_v_heads                 48      32   qwen4exp.ssm.time_step_rank
//   ssm_value_dim             6144    4096   qwen4exp.ssm.inner_size
//   idx_top_k                 2048  262144   qwen4exp.attention.indexer.top_k
// ---------------------------------------------------------------------------------------------------

// Compile-time profile: a define of the bare token (flashnext / w45) is turned into a comparison chain.
// The preprocessor cannot compare identifiers, so the profiles are selected with a small #if ladder over
// user-defined macros the CMake sets when the profile is NOT the default.
#if defined(STRATA_GEOM_PROFILE_W45)

// ---- Whittle-Qwen-3.8-45B-A3B: a Qwen3.6-35B-A3B (qwen4exp) with 76 restored experts ----
#ifndef STRATA_N_EMBD
#define STRATA_N_EMBD 2048
#endif
#ifndef STRATA_N_LAYERS
#define STRATA_N_LAYERS 40
#endif
#ifndef STRATA_N_FF
#define STRATA_N_FF 512
#endif
#ifndef STRATA_N_EXPERT
#define STRATA_N_EXPERT 256
#endif
#ifndef STRATA_N_EXPERT_USED
#define STRATA_N_EXPERT_USED 8
#endif
#ifndef STRATA_N_HEAD
#define STRATA_N_HEAD 16
#endif
#ifndef STRATA_N_HEAD_KV
#define STRATA_N_HEAD_KV 2
#endif
#ifndef STRATA_SSM_V_HEADS
#define STRATA_SSM_V_HEADS 32
#endif
#ifndef STRATA_SSM_VALUE_DIM
#define STRATA_SSM_VALUE_DIM 4096
#endif
#ifndef STRATA_IDX_TOP_K
#define STRATA_IDX_TOP_K 262144
#endif

#else

// ---- default: Qwen3.8-Flash-Next (upstream's shape) ----
#ifndef STRATA_N_EMBD
#define STRATA_N_EMBD 2560
#endif
#ifndef STRATA_N_LAYERS
#define STRATA_N_LAYERS 48
#endif
#ifndef STRATA_N_FF
#define STRATA_N_FF 640
#endif
#ifndef STRATA_N_EXPERT
#define STRATA_N_EXPERT 512
#endif
#ifndef STRATA_N_EXPERT_USED
#define STRATA_N_EXPERT_USED 10
#endif
#ifndef STRATA_N_HEAD
#define STRATA_N_HEAD 24
#endif
#ifndef STRATA_N_HEAD_KV
#define STRATA_N_HEAD_KV 2
#endif
#ifndef STRATA_SSM_V_HEADS
#define STRATA_SSM_V_HEADS 48
#endif
#ifndef STRATA_SSM_VALUE_DIM
#define STRATA_SSM_VALUE_DIM 6144
#endif
#ifndef STRATA_IDX_TOP_K
#define STRATA_IDX_TOP_K 2048
#endif

#endif  // profile

// ---------------------------------------------------------------------------------------------------
// Axes shared by both profiles (verified equal on Whittle and Flash-Next: hyper-connections, the
// indexer's head/key geometry, the attention head width, the GDN convolution and state size)
// ---------------------------------------------------------------------------------------------------
#ifndef STRATA_HC
#define STRATA_HC 4
#endif
#ifndef STRATA_HC_LR
#define STRATA_HC_LR 320
#endif
#ifndef STRATA_HEAD_DIM
#define STRATA_HEAD_DIM 256
#endif
#ifndef STRATA_N_ROT
#define STRATA_N_ROT 64
#endif
#ifndef STRATA_IDX_Q_HEADS
#define STRATA_IDX_Q_HEADS 4
#endif
#ifndef STRATA_IDX_KEY_DIM
#define STRATA_IDX_KEY_DIM 128
#endif
#ifndef STRATA_IDX_BLOCK
#define STRATA_IDX_BLOCK 4
#endif
#ifndef STRATA_SSM_STATE_SIZE
#define STRATA_SSM_STATE_SIZE 128
#endif
#ifndef STRATA_SSM_K_HEADS
#define STRATA_SSM_K_HEADS 16
#endif
#ifndef STRATA_SSM_D_CONV
#define STRATA_SSM_D_CONV 4
#endif
#ifndef STRATA_QSA_INTERVAL
#define STRATA_QSA_INTERVAL 4
#endif

// ---------------------------------------------------------------------------------------------------
// Derived values the kernels use in place of the literals they used to carry
// ---------------------------------------------------------------------------------------------------

/// hc * n_embd: the gated-residual stream width (10240 upstream, 8192 on w45)
#define STRATA_HC_DIM (STRATA_HC * STRATA_N_EMBD)

/// The SSM mixers' convolution channel count: 2 * state * k_heads + value_dim (10240 / 8192)
#ifndef STRATA_SSM_CONV_CHANNELS
#define STRATA_SSM_CONV_CHANNELS \
    (2 * STRATA_SSM_STATE_SIZE * STRATA_SSM_K_HEADS + STRATA_SSM_VALUE_DIM)
#endif

/// One expert's packed blob, in bytes, for a Q2_0-style 18-bytes-per-64-weights layout:
/// 3 roles * n_embd * n_ff * 18 / 64  ->  1,382,400 upstream, 884,736 on w45.
/// (The native pack formats derive their own sizes from the file; this is the canonical Q2_0 arena.)
#ifndef STRATA_EXPERT_BLOB
#define STRATA_EXPERT_BLOB (3u * STRATA_N_EMBD * STRATA_N_FF * 18u / 64u)
#endif

/// Stringify a macro value (for build messages that name a compiled width)
#define STRATA_STR_(x) #x
#define STRATA_STR(x) STRATA_STR_(x)

/// Q8_1 blocks of one activation row, as the S26 kernels index them
#define STRATA_S26_XMAX (STRATA_N_EMBD / 32)
