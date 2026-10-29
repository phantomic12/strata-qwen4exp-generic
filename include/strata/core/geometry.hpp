// include/strata/core/geometry.hpp - the model's geometry: one struct for the shape a model file declares and
// the shape the kernels are compiled for.
//
// A model file's geometry is read from its own metadata (`strata::read_geometry`, gguf_reader.hpp), and the
// engine passes that struct everywhere it needs a dimension.  The CUDA/HIP kernels still fix most of these
// numbers at compile time (`constexpr int N = 2560` and the like); `kCompiledGeometry` is that shape, written
// once, and each kernel that pins a number `static_assert`s it against this struct.  So a kernel and this file
// cannot drift apart without the build failing, and a model file of another shape is refused at load with the
// metadata key that differs (`strata::geometry_unsupported`) instead of mis-running.
//
// Dependency-free on purpose: the kernels include it.
#pragma once

#include "strata/artifact/model_family.hpp"
#include "strata/core/geometry_build.hpp"

#include <cstdint>

namespace strata::core {

/// Every field is a number a kernel depends on, so a change to the compiled values is a change to a kernel
/// contract and not a tuning knob.  The defaults are Qwen3.8-Flash-Next's (qwen4exp), the shape the kernels
/// are compiled for; the GGUF key each field is read from is named beside it.
struct ModelGeometry {
    int64_t n_embd = STRATA_N_EMBD;          ///< embedding_length
    int64_t n_layers = STRATA_N_LAYERS;          ///< block_count (the trunk; an in-file MTP block is counted in n_mtp_layers)
    int64_t qsa_interval = STRATA_QSA_INTERVAL;       ///< full_attention_interval: every 4th layer is full attention, 3, 7, ... 47
    int64_t n_vocab = 248320;       ///< the length of tokenizer.ggml.tokens
    double rms_eps = 1e-6;          ///< attention.layer_norm_rms_epsilon

    // GDN (36 layers)
    int64_t ssm_state_size = STRATA_SSM_STATE_SIZE;        ///< ssm.state_size: the key head width, and the value head width
    int64_t ssm_k_heads = STRATA_SSM_K_HEADS;            ///< ssm.group_count
    int64_t ssm_v_heads = STRATA_SSM_V_HEADS;            ///< ssm.time_step_rank
    int64_t ssm_d_conv = STRATA_SSM_D_CONV;              ///< ssm.conv_kernel
    int64_t ssm_conv_channels = STRATA_SSM_CONV_CHANNELS;   ///< 2*128*16 + 128*48, derived
    int64_t ssm_value_dim = STRATA_SSM_VALUE_DIM;        ///< ssm.inner_size, 128 * 48

    // QSA (12 layers)
    int64_t n_head = STRATA_N_HEAD;            ///< attention.head_count
    int64_t n_head_kv = STRATA_N_HEAD_KV;          ///< attention.head_count_kv
    int64_t head_dim = STRATA_HEAD_DIM;         ///< attention.key_length (= attention.value_length)
    int64_t n_rot = STRATA_N_ROT;             ///< rope.dimension_count: the rotated slice of head_dim and of idx_key_dim
    int64_t idx_q_heads = STRATA_IDX_Q_HEADS;        ///< attention.indexer.head_count
    int64_t idx_key_dim = STRATA_IDX_KEY_DIM;      ///< attention.indexer.key_length
    int64_t idx_block = STRATA_IDX_BLOCK;          ///< the non-zero attention.compress_ratios entry: one pooled key per 4 cells
    int64_t idx_top_k = STRATA_IDX_TOP_K;       ///< attention.indexer.top_k: cells of selection budget

    // gated residual, on every layer
    int64_t hc = STRATA_HC;                 ///< hyper_connection.count
    int64_t hc_lr = STRATA_HC_LR;            ///< hyper_connection.low_rank

    // MoE, on every layer
    int64_t n_expert = STRATA_N_EXPERT;         ///< expert_count; pruned variants (GSQ-RCO Coder) ship fewer
    int64_t n_expert_used = STRATA_N_EXPERT_USED;     ///< expert_used_count
    int64_t n_ff = STRATA_N_FF;             ///< expert_feed_forward_length (= expert_shared_feed_forward_length)

    // appended for the model families (Qwen3.6 = qwen35moe); the defaults keep Flash-Next's values
    int64_t n_ff_shexp = STRATA_N_FF;       ///< the shared expert's FFN width (qwen4exp: equal to n_ff, read_geometry checks)
    ModelFamily family = ModelFamily::Qwen4Exp;
    int64_t n_mtp_layers = 0;       ///< MTP blocks inside the model file, after the trunk (qwen35moe: blk.40)

    constexpr bool has_indexer() const { return idx_q_heads > 0; }              ///< false: dense attention only
    constexpr bool has_hyper_connections() const { return hc > 0; }             ///< false: plain pre-norm residual
    constexpr int64_t hc_dim() const { return hc * n_embd; }
    /// `layer % qsa_interval == qsa_interval - 1` is full attention.  Derived, not a second list.
    constexpr int64_t n_qsa_layers() const { return n_layers / qsa_interval; }
    constexpr int64_t n_gdn_layers() const { return n_layers - n_qsa_layers(); }
};

/// The shape the kernels are compiled for.  A kernel that fixes a dimension at compile time asserts it here.
inline constexpr ModelGeometry kCompiledGeometry{};

/// True for the full-attention layers.  `docs/semantics.md` gives this twice over - `full_attention_interval
/// = 4` and an explicit `attention.compress_ratios` array - and `read_geometry` checks that the two agree.
inline bool is_qsa_layer(const ModelGeometry& g, int64_t layer) {
    return layer % g.qsa_interval == g.qsa_interval - 1;
}

}  // namespace strata::core
