/*
 * model.h — Model structures and loading for the LLM-MoE runtime
 *
 * Core types:
 *   Config         — model hyperparameters
 *   Tensor         — weight tensor (FP32 or INT4)
 *   KVCache        — per-layer key/value cache
 *   ExpertMLP      — single MoE expert
 *   MoELayer       — full MoE with gating
 *   TransformerLayer — attention + MoE block
 *   Model          — complete model
 *   RunState       — inference buffers
 */

#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Quantization Types ─────────────────────────────────────────────── */
#define QUANT_NONE  0   /* FP32 */
#define QUANT_INT4  1   /* INT4 packed, 2 per byte */

/* ── Config ─────────────────────────────────────────────────────────── */
typedef struct {
    uint32_t vocab_size;
    uint32_t max_seq_len;
    uint32_t dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t n_experts;
    uint32_t n_active_experts;
    uint32_t expert_dim;
    float    norm_eps;
    float    rope_theta;
    uint32_t pad_id;
    uint32_t unk_id;
    uint32_t bos_id;
    uint32_t eos_id;
} Config;

/* ── Tensor ─────────────────────────────────────────────────────────── */
typedef struct {
    uint8_t  quant_type;        /* QUANT_NONE or QUANT_INT4 */
    uint8_t  ndims;
    uint32_t shape[4];          /* up to 4 dimensions */
    uint32_t numel;             /* total number of elements */
    float    scale;             /* for INT4: dequant scale */
    union {
        float   *fp32;          /* QUANT_NONE data */
        uint8_t *int4_packed;   /* QUANT_INT4 data */
    } data;
} Tensor;

/* ── KV Cache ───────────────────────────────────────────────────────── */
typedef struct {
    float *key_cache;    /* [max_seq_len, n_heads, head_dim] */
    float *value_cache;  /* [max_seq_len, n_heads, head_dim] */
    int    length;       /* current number of cached positions */
} KVCache;

/* ── Expert MLP ─────────────────────────────────────────────────────── */
typedef struct {
    Tensor w1;   /* gate projection:  [dim, expert_dim] */
    Tensor w2;   /* down projection:  [expert_dim, dim] */
    Tensor w3;   /* up projection:    [dim, expert_dim] */
} ExpertMLP;

/* ── MoE Layer ──────────────────────────────────────────────────────── */
typedef struct {
    Tensor    gate;      /* gating weights: [dim, n_experts] */
    ExpertMLP *experts;  /* array of n_experts */
} MoELayer;

/* ── Transformer Layer ──────────────────────────────────────────────── */
typedef struct {
    /* Attention */
    Tensor attn_norm;    /* RMSNorm weight: [dim] */
    Tensor wq;           /* query projection:  [dim, dim] */
    Tensor wk;           /* key projection:    [dim, dim] */
    Tensor wv;           /* value projection:  [dim, dim] */
    Tensor wo;           /* output projection: [dim, dim] */

    /* MoE */
    Tensor   ffn_norm;   /* RMSNorm weight: [dim] */
    MoELayer moe;
} TransformerLayer;

/* ── Model ──────────────────────────────────────────────────────────── */
typedef struct {
    Config            config;
    Tensor            embedding;   /* [vocab_size, dim] */
    TransformerLayer *layers;      /* array of n_layers */
    Tensor            final_norm;  /* [dim] */
    Tensor            output;      /* [vocab_size, dim] — may alias embedding */
    int               weight_tied; /* 1 if output shares embedding weights */
} Model;

/* ── RunState (inference buffers) ───────────────────────────────────── */
typedef struct {
    float *x;          /* current hidden state:  [dim] */
    float *xb;         /* buffer after norm:     [dim] */
    float *xb2;        /* second buffer:         [dim] */
    float *q;          /* query:     [n_heads * head_dim] */
    float *k;          /* key:       [n_heads * head_dim] */
    float *v;          /* value:     [n_heads * head_dim] */
    float *att;        /* attention scores: [n_heads * max_seq_len] */
    float *hb;         /* expert hidden: [expert_dim] */
    float *hb2;        /* expert hidden2: [expert_dim] */
    float *hb3;        /* expert hidden3: [expert_dim] */
    float *expert_out; /* single expert output: [dim] */
    float *moe_out;    /* combined MoE output: [dim] */
    float *gate_logits;/* gating logits: [n_experts] */
    float *logits;     /* output logits: [vocab_size] */

    KVCache *kv_caches;  /* per-layer KV caches */
} RunState;

/* ── API ────────────────────────────────────────────────────────────── */

/* Load model from binary file. Returns 0 on success. */
int model_load(Model *model, const char *filepath);

/* Free all model memory. */
void model_free(Model *model);

/* Allocate inference buffers. Returns 0 on success. */
int runstate_alloc(RunState *state, const Config *config);

/* Free inference buffers. */
void runstate_free(RunState *state);

/* Run a single forward pass for one token.
 * token: input token ID
 * pos:   position in sequence
 * Returns pointer to logits [vocab_size].
 */
float *forward(Model *model, RunState *state, int token, int pos);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_H */
