/*
 * model.c — Model loading and forward pass
 *
 * Loads the full model from the custom binary format.
 * Implements the complete forward pass:
 *   embed → [RMSNorm → Attention → RMSNorm → MoE] × N → RMSNorm → Output
 */

#include "model.h"
#include "math_ops.h"
#include "quantize.h"
#include "attention.h"
#include "moe.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Tensor Loading
 * ═══════════════════════════════════════════════════════════════════════*/

static int load_tensor(FILE *fp, Tensor *t)
{
    /* Name (skip) */
    uint16_t name_len;
    if (fread(&name_len, sizeof(uint16_t), 1, fp) != 1) return -1;
    if (fseek(fp, name_len, SEEK_CUR) != 0) return -1;

    /* Quant type */
    uint8_t qtype;
    if (fread(&qtype, sizeof(uint8_t), 1, fp) != 1) return -1;
    t->quant_type = qtype;

    /* Shape */
    uint8_t ndims;
    if (fread(&ndims, sizeof(uint8_t), 1, fp) != 1) return -1;
    t->ndims = ndims;

    t->numel = 1;
    for (int i = 0; i < ndims; i++) {
        uint32_t s;
        if (fread(&s, sizeof(uint32_t), 1, fp) != 1) return -1;
        t->shape[i] = s;
        t->numel *= s;
    }

    if (qtype == QUANT_INT4) {
        /* Scale */
        if (fread(&t->scale, sizeof(float), 1, fp) != 1) return -1;

        /* Number of elements */
        uint32_t numel;
        if (fread(&numel, sizeof(uint32_t), 1, fp) != 1) return -1;
        t->numel = numel;

        /* Packed data */
        size_t packed_size = ((size_t)numel + 1) / 2;
        t->data.int4_packed = (uint8_t *)malloc(packed_size);
        if (!t->data.int4_packed) return -1;
        if (fread(t->data.int4_packed, 1, packed_size, fp) != packed_size) return -1;
    } else {
        /* FP32 data */
        t->scale = 1.0f;
        t->data.fp32 = (float *)malloc((size_t)t->numel * sizeof(float));
        if (!t->data.fp32) return -1;
        if (fread(t->data.fp32, sizeof(float), t->numel, fp) != t->numel) return -1;
    }

    return 0;
}

static void free_tensor(Tensor *t)
{
    if (t->quant_type == QUANT_INT4) {
        free(t->data.int4_packed);
        t->data.int4_packed = NULL;
    } else {
        free(t->data.fp32);
        t->data.fp32 = NULL;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Model Loading
 *
 * Binary format:
 *   [MAGIC "LLMO" 4B] [VERSION 4B]
 *   [CONFIG: 15 fields]
 *   [TOKENIZER: vocab + merges]
 *   [N_TENSORS 4B] [Tensor × N]
 * ═══════════════════════════════════════════════════════════════════════*/

/* Map tensor names to model fields */
typedef struct {
    char     name[128];
    Tensor  *dest;
} TensorMapping;

int model_load(Model *model, const char *filepath)
{
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s\n", filepath);
        return -1;
    }

    /* ── Magic ── */
    char magic[4];
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "LLMO", 4) != 0) {
        fprintf(stderr, "Error: invalid magic number\n");
        fclose(fp);
        return -1;
    }

    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return -1; }
    fprintf(stderr, "Model format version: %u\n", version);

    /* ── Config ── */
    Config *cfg = &model->config;
    fread(&cfg->vocab_size, sizeof(uint32_t), 1, fp);
    fread(&cfg->max_seq_len, sizeof(uint32_t), 1, fp);
    fread(&cfg->dim, sizeof(uint32_t), 1, fp);
    fread(&cfg->n_layers, sizeof(uint32_t), 1, fp);
    fread(&cfg->n_heads, sizeof(uint32_t), 1, fp);
    fread(&cfg->head_dim, sizeof(uint32_t), 1, fp);
    fread(&cfg->n_experts, sizeof(uint32_t), 1, fp);
    fread(&cfg->n_active_experts, sizeof(uint32_t), 1, fp);
    fread(&cfg->expert_dim, sizeof(uint32_t), 1, fp);
    fread(&cfg->norm_eps, sizeof(float), 1, fp);
    fread(&cfg->rope_theta, sizeof(float), 1, fp);
    fread(&cfg->pad_id, sizeof(uint32_t), 1, fp);
    fread(&cfg->unk_id, sizeof(uint32_t), 1, fp);
    fread(&cfg->bos_id, sizeof(uint32_t), 1, fp);
    fread(&cfg->eos_id, sizeof(uint32_t), 1, fp);

    fprintf(stderr, "Config: dim=%u, layers=%u, heads=%u, experts=%u (top-%u), "
            "expert_dim=%u, vocab=%u, max_seq=%u\n",
            cfg->dim, cfg->n_layers, cfg->n_heads, cfg->n_experts,
            cfg->n_active_experts, cfg->expert_dim, cfg->vocab_size,
            cfg->max_seq_len);

    /* ── Tokenizer (skip — loaded separately) ── */
    /* We need to read past the tokenizer section */
    uint32_t tok_vocab_size, tok_n_merges;
    fread(&tok_vocab_size, sizeof(uint32_t), 1, fp);
    fread(&tok_n_merges, sizeof(uint32_t), 1, fp);

    /* Skip vocab entries */
    for (uint32_t i = 0; i < tok_vocab_size; i++) {
        uint16_t len;
        fread(&len, sizeof(uint16_t), 1, fp);
        fseek(fp, len + sizeof(uint32_t), SEEK_CUR);  /* bytes + token_id */
    }

    /* Skip merge rules */
    for (uint32_t i = 0; i < tok_n_merges; i++) {
        uint16_t len;
        fread(&len, sizeof(uint16_t), 1, fp);
        fseek(fp, len, SEEK_CUR);
        fread(&len, sizeof(uint16_t), 1, fp);
        fseek(fp, len, SEEK_CUR);
    }

    /* ── Allocate layers ── */
    model->layers = (TransformerLayer *)calloc(cfg->n_layers, sizeof(TransformerLayer));
    if (!model->layers) { fclose(fp); return -1; }

    for (uint32_t l = 0; l < cfg->n_layers; l++) {
        model->layers[l].moe.experts = (ExpertMLP *)calloc(
            cfg->n_experts, sizeof(ExpertMLP));
        if (!model->layers[l].moe.experts) { fclose(fp); return -1; }
    }

    /* ── Build tensor name → destination mapping ── */
    /* We'll match tensor names as they come */
    uint32_t n_tensors;
    fread(&n_tensors, sizeof(uint32_t), 1, fp);
    fprintf(stderr, "Loading %u tensors...\n", n_tensors);

    for (uint32_t t = 0; t < n_tensors; t++) {
        /* Peek at tensor name */
        long pos = ftell(fp);

        uint16_t name_len;
        fread(&name_len, sizeof(uint16_t), 1, fp);

        char name[256];
        if (name_len >= sizeof(name)) name_len = sizeof(name) - 1;
        fread(name, 1, name_len, fp);
        name[name_len] = '\0';

        /* Seek back to start of tensor */
        fseek(fp, pos, SEEK_SET);

        /* Determine destination */
        Tensor *dest = NULL;

        /* Parse layer index and field name */
        int layer_idx;
        int expert_idx;
        char field[128];

        if (strcmp(name, "embedding.weight") == 0) {
            dest = &model->embedding;
        } else if (strcmp(name, "norm.weight") == 0) {
            dest = &model->final_norm;
        } else if (strcmp(name, "output.weight") == 0) {
            dest = &model->output;
            model->weight_tied = 0;  /* explicit output weight */
        } else if (sscanf(name, "layers.%d.attn_norm.weight", &layer_idx) == 1) {
            dest = &model->layers[layer_idx].attn_norm;
        } else if (sscanf(name, "layers.%d.attn.wq.weight", &layer_idx) == 1) {
            dest = &model->layers[layer_idx].wq;
        } else if (sscanf(name, "layers.%d.attn.wk.weight", &layer_idx) == 1) {
            dest = &model->layers[layer_idx].wk;
        } else if (sscanf(name, "layers.%d.attn.wv.weight", &layer_idx) == 1) {
            dest = &model->layers[layer_idx].wv;
        } else if (sscanf(name, "layers.%d.attn.wo.weight", &layer_idx) == 1) {
            dest = &model->layers[layer_idx].wo;
        } else if (sscanf(name, "layers.%d.ffn_norm.weight", &layer_idx) == 1) {
            dest = &model->layers[layer_idx].ffn_norm;
        } else if (sscanf(name, "layers.%d.moe.gate.weight", &layer_idx) == 1) {
            dest = &model->layers[layer_idx].moe.gate;
        } else if (sscanf(name, "layers.%d.moe.experts.%d.w1.weight",
                          &layer_idx, &expert_idx) == 2) {
            dest = &model->layers[layer_idx].moe.experts[expert_idx].w1;
        } else if (sscanf(name, "layers.%d.moe.experts.%d.w2.weight",
                          &layer_idx, &expert_idx) == 2) {
            dest = &model->layers[layer_idx].moe.experts[expert_idx].w2;
        } else if (sscanf(name, "layers.%d.moe.experts.%d.w3.weight",
                          &layer_idx, &expert_idx) == 2) {
            dest = &model->layers[layer_idx].moe.experts[expert_idx].w3;
        } else {
            fprintf(stderr, "  Warning: unknown tensor '%s', skipping\n", name);
        }

        if (dest) {
            if (load_tensor(fp, dest) != 0) {
                fprintf(stderr, "Error loading tensor '%s'\n", name);
                fclose(fp);
                return -1;
            }
        } else {
            /* Skip unknown tensor */
            load_tensor(fp, &(Tensor){0});
        }
    }

    /* Handle weight tying: if output.weight was not loaded, alias embedding */
    if (model->output.data.fp32 == NULL && model->output.data.int4_packed == NULL) {
        model->output = model->embedding;
        model->weight_tied = 1;
        fprintf(stderr, "  Weight tying: output = embedding\n");
    }

    fclose(fp);
    fprintf(stderr, "Model loaded successfully.\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Memory Management
 * ═══════════════════════════════════════════════════════════════════════*/

void model_free(Model *model)
{
    free_tensor(&model->embedding);
    free_tensor(&model->final_norm);
    if (!model->weight_tied) {
        free_tensor(&model->output);
    }

    if (model->layers) {
        for (uint32_t l = 0; l < model->config.n_layers; l++) {
            TransformerLayer *layer = &model->layers[l];
            free_tensor(&layer->attn_norm);
            free_tensor(&layer->wq);
            free_tensor(&layer->wk);
            free_tensor(&layer->wv);
            free_tensor(&layer->wo);
            free_tensor(&layer->ffn_norm);
            free_tensor(&layer->moe.gate);

            if (layer->moe.experts) {
                for (uint32_t e = 0; e < model->config.n_experts; e++) {
                    free_tensor(&layer->moe.experts[e].w1);
                    free_tensor(&layer->moe.experts[e].w2);
                    free_tensor(&layer->moe.experts[e].w3);
                }
                free(layer->moe.experts);
            }
        }
        free(model->layers);
    }
}

int runstate_alloc(RunState *state, const Config *config)
{
    int dim        = (int)config->dim;
    int n_heads    = (int)config->n_heads;
    int head_dim   = (int)config->head_dim;
    int max_seq    = (int)config->max_seq_len;
    int n_experts  = (int)config->n_experts;
    int expert_dim = (int)config->expert_dim;
    int vocab_size = (int)config->vocab_size;
    int n_layers   = (int)config->n_layers;
    int kv_dim     = n_heads * head_dim;

    state->x          = (float *)calloc((size_t)dim, sizeof(float));
    state->xb         = (float *)calloc((size_t)dim, sizeof(float));
    state->xb2        = (float *)calloc((size_t)dim, sizeof(float));
    state->q          = (float *)calloc((size_t)kv_dim, sizeof(float));
    state->k          = (float *)calloc((size_t)kv_dim, sizeof(float));
    state->v          = (float *)calloc((size_t)kv_dim, sizeof(float));
    state->att        = (float *)calloc((size_t)(n_heads * max_seq), sizeof(float));
    state->hb         = (float *)calloc((size_t)expert_dim, sizeof(float));
    state->hb2        = (float *)calloc((size_t)expert_dim, sizeof(float));
    state->hb3        = (float *)calloc((size_t)expert_dim, sizeof(float));
    state->expert_out = (float *)calloc((size_t)dim, sizeof(float));
    state->moe_out    = (float *)calloc((size_t)dim, sizeof(float));
    state->gate_logits= (float *)calloc((size_t)n_experts, sizeof(float));
    state->logits     = (float *)calloc((size_t)vocab_size, sizeof(float));

    /* KV caches */
    state->kv_caches = (KVCache *)calloc((size_t)n_layers, sizeof(KVCache));
    if (!state->kv_caches) return -1;

    for (int l = 0; l < n_layers; l++) {
        size_t cache_size = (size_t)max_seq * kv_dim;
        state->kv_caches[l].key_cache   = (float *)calloc(cache_size, sizeof(float));
        state->kv_caches[l].value_cache = (float *)calloc(cache_size, sizeof(float));
        state->kv_caches[l].length = 0;

        if (!state->kv_caches[l].key_cache || !state->kv_caches[l].value_cache)
            return -1;
    }

    /* Verify all allocations */
    if (!state->x || !state->xb || !state->xb2 || !state->q ||
        !state->k || !state->v || !state->att || !state->hb ||
        !state->hb2 || !state->hb3 || !state->expert_out ||
        !state->moe_out || !state->gate_logits || !state->logits)
        return -1;

    return 0;
}

void runstate_free(RunState *state)
{
    free(state->x);
    free(state->xb);
    free(state->xb2);
    free(state->q);
    free(state->k);
    free(state->v);
    free(state->att);
    free(state->hb);
    free(state->hb2);
    free(state->hb3);
    free(state->expert_out);
    free(state->moe_out);
    free(state->gate_logits);
    free(state->logits);

    if (state->kv_caches) {
        /* We don't know n_layers here, free until NULL */
        /* Actually we stored them contiguously, just free each */
        /* This is a simplification — in production, store n_layers */
        free(state->kv_caches);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Helper: matmul dispatching on tensor quant type
 * ═══════════════════════════════════════════════════════════════════════*/

static void tensor_matvec(float *out, const float *in,
                          const Tensor *w, int in_dim, int out_dim)
{
    /* w is [out_dim, in_dim] stored row-major
     * out[i] = sum_j(w[i][j] * in[j])
     * This is matmul with M=1 */
    if (w->quant_type == QUANT_INT4) {
        matmul_int4(out, in, w->data.int4_packed, w->scale, 1, in_dim, out_dim);
    } else {
        matmul(out, in, w->data.fp32, 1, in_dim, out_dim);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Forward Pass (single token)
 * ═══════════════════════════════════════════════════════════════════════*/

float *forward(Model *model, RunState *state, int token, int pos)
{
    Config *cfg = &model->config;
    int dim      = (int)cfg->dim;
    int n_heads  = (int)cfg->n_heads;
    int head_dim = (int)cfg->head_dim;
    int kv_dim   = n_heads * head_dim;

    /* 1. Token embedding: x = embedding[token] */
    if (model->embedding.quant_type == QUANT_INT4) {
        /* Dequantize the embedding row */
        int offset = token * dim;
        dequantize_int4(state->x, model->embedding.data.int4_packed + offset / 2,
                        model->embedding.scale, dim);
    } else {
        float *emb_row = model->embedding.data.fp32 + token * dim;
        vec_copy(state->x, emb_row, dim);
    }

    /* 2. Transformer layers */
    for (uint32_t l = 0; l < cfg->n_layers; l++) {
        TransformerLayer *layer = &model->layers[l];

        /* ── Pre-attention RMSNorm ── */
        rmsnorm(state->xb, state->x, layer->attn_norm.data.fp32,
                dim, cfg->norm_eps);

        /* ── QKV Projections ── */
        tensor_matvec(state->q, state->xb, &layer->wq, dim, kv_dim);
        tensor_matvec(state->k, state->xb, &layer->wk, dim, kv_dim);
        tensor_matvec(state->v, state->xb, &layer->wv, dim, kv_dim);

        /* ── Apply RoPE to Q and K ── */
        for (int h = 0; h < n_heads; h++) {
            rope(state->q + h * head_dim,
                 state->k + h * head_dim,
                 head_dim, pos, cfg->rope_theta);
        }

        /* ── Attention with KV Cache ── */
        attention_forward(state, &state->kv_caches[l], cfg, pos);

        /* ── Output projection: xb2 = Wo · xb ── */
        tensor_matvec(state->xb2, state->xb, &layer->wo, kv_dim, dim);

        /* ── Residual connection ── */
        vec_add(state->x, state->x, state->xb2, dim);

        /* ── Pre-MoE RMSNorm ── */
        rmsnorm(state->xb, state->x, layer->ffn_norm.data.fp32,
                dim, cfg->norm_eps);

        /* ── MoE Forward ── */
        moe_forward(state->moe_out, state->xb, &layer->moe, state, cfg);

        /* ── Residual connection ── */
        vec_add(state->x, state->x, state->moe_out, dim);
    }

    /* 3. Final RMSNorm */
    rmsnorm(state->x, state->x, model->final_norm.data.fp32,
            dim, cfg->norm_eps);

    /* 4. Output projection: logits = x · output^T */
    tensor_matvec(state->logits, state->x, &model->output,
                  dim, (int)cfg->vocab_size);

    return state->logits;
}
