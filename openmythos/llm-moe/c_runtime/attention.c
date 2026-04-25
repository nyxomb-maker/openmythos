/*
 * attention.c — Causal self-attention implementation
 *
 * Two modes:
 *   1. Flash Attention (streaming) — default, O(1) memory per head
 *   2. Standard attention — for validation, O(N) memory per head
 *
 * Both use KV cache for O(1) per-token inference.
 */

#include "attention.h"
#include "math_ops.h"
#include <math.h>
#include <float.h>
#include <string.h>

/* Flash Attention block size */
#define FA_BLOCK_SIZE 64

/* ═══════════════════════════════════════════════════════════════════════
 * Flash Attention (Streaming)
 *
 * For each query head:
 *   Process cached K/V in blocks of FA_BLOCK_SIZE.
 *   Maintain running statistics (max, sum_exp, output) using
 *   the online softmax trick:
 *
 *   For each k[t]:
 *     score = dot(q, k[t]) / sqrt(d)
 *     new_max = max(running_max, score)
 *     correction = exp(running_max - new_max)
 *     running_sum = running_sum * correction + exp(score - new_max)
 *     output = output * correction + exp(score - new_max) * v[t]
 *     running_max = new_max
 *
 *   Final: output = output / running_sum
 *
 * This avoids materializing the N×N attention matrix.
 * ═══════════════════════════════════════════════════════════════════════*/

void attention_forward(RunState *state, KVCache *cache,
                       const Config *config, int pos)
{
    int dim      = (int)config->dim;
    int n_heads  = (int)config->n_heads;
    int head_dim = (int)config->head_dim;
    int kv_len   = pos + 1;  /* includes current token */
    float scale  = 1.0f / sqrtf((float)head_dim);

    /* 1. Store current K and V into cache */
    int cache_offset = pos * n_heads * head_dim;
    memcpy(cache->key_cache + cache_offset, state->k,
           (size_t)(n_heads * head_dim) * sizeof(float));
    memcpy(cache->value_cache + cache_offset, state->v,
           (size_t)(n_heads * head_dim) * sizeof(float));
    cache->length = kv_len;

    /* 2. For each attention head, compute attention using streaming */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < n_heads; h++) {
        float *q_head = state->q + h * head_dim;
        float *out    = state->xb + h * head_dim;  /* output per head */

        /* Initialize streaming state */
        float running_max = -FLT_MAX;
        float running_sum = 0.0f;

        /* Zero the output accumulator */
        for (int d = 0; d < head_dim; d++) {
            out[d] = 0.0f;
        }

        /* Process all cached positions */
        for (int t = 0; t < kv_len; t++) {
            float *k_t = cache->key_cache + t * n_heads * head_dim + h * head_dim;
            float *v_t = cache->value_cache + t * n_heads * head_dim + h * head_dim;

            /* Compute attention score */
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_head[d] * k_t[d];
            }
            score *= scale;

            /* Online softmax update */
            float new_max = (score > running_max) ? score : running_max;
            float correction = expf(running_max - new_max);
            float exp_score = expf(score - new_max);

            /* Update running statistics */
            running_sum = running_sum * correction + exp_score;

            /* Update output: rescale old output + add new contribution */
            for (int d = 0; d < head_dim; d++) {
                out[d] = out[d] * correction + exp_score * v_t[d];
            }

            running_max = new_max;
        }

        /* Normalize */
        if (running_sum > 0.0f) {
            float inv_sum = 1.0f / running_sum;
            for (int d = 0; d < head_dim; d++) {
                out[d] *= inv_sum;
            }
        }
    }

    /* state->xb now contains the concatenated attention output [dim] */
}

/* ═══════════════════════════════════════════════════════════════════════
 * Standard Attention (for validation)
 *
 * Materializes full attention scores per head.
 * Uses state->att as scratch buffer [n_heads * max_seq_len].
 * ═══════════════════════════════════════════════════════════════════════*/

void attention_forward_standard(RunState *state, KVCache *cache,
                                const Config *config, int pos)
{
    int dim      = (int)config->dim;
    int n_heads  = (int)config->n_heads;
    int head_dim = (int)config->head_dim;
    int kv_len   = pos + 1;
    int max_seq  = (int)config->max_seq_len;
    float scale  = 1.0f / sqrtf((float)head_dim);

    /* Store current K and V into cache */
    int cache_offset = pos * n_heads * head_dim;
    memcpy(cache->key_cache + cache_offset, state->k,
           (size_t)(n_heads * head_dim) * sizeof(float));
    memcpy(cache->value_cache + cache_offset, state->v,
           (size_t)(n_heads * head_dim) * sizeof(float));
    cache->length = kv_len;

    /* For each head */
    for (int h = 0; h < n_heads; h++) {
        float *q_head = state->q + h * head_dim;
        float *att = state->att + h * max_seq;

        /* Compute attention scores for all cached positions */
        for (int t = 0; t < kv_len; t++) {
            float *k_t = cache->key_cache + t * n_heads * head_dim + h * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_head[d] * k_t[d];
            }
            att[t] = score * scale;
        }

        /* Softmax over attention scores */
        softmax(att, kv_len);

        /* Weighted sum of values */
        float *out = state->xb + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            out[d] = 0.0f;
        }
        for (int t = 0; t < kv_len; t++) {
            float *v_t = cache->value_cache + t * n_heads * head_dim + h * head_dim;
            float w = att[t];
            for (int d = 0; d < head_dim; d++) {
                out[d] += w * v_t[d];
            }
        }
    }
}
