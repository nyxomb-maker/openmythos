/*
 * moe.c — Mixture of Experts implementation
 *
 * Each token is routed to the top-K experts based on a learned gate.
 * Each expert is a SwiGLU MLP: out = W2(SiLU(W1·x) * W3·x)
 * Final output is the weighted combination of expert outputs.
 */

#include "moe.h"
#include "math_ops.h"
#include "quantize.h"
#include <string.h>
#include <float.h>

/* ── Helper: compute matmul dispatching on quant type ───────────────── */
static void tensor_matmul(float *out, const float *in,
                          const Tensor *w, int M, int K, int N)
{
    if (w->quant_type == QUANT_INT4) {
        matmul_int4(out, in, w->data.int4_packed, w->scale, M, K, N);
    } else {
        matmul(out, in, w->data.fp32, M, K, N);
    }
}

/* ── Helper: find top-K indices and values ──────────────────────────── */
static void topk(const float *probs, int n, int k,
                 int *indices, float *values)
{
    /* Simple selection sort for small k */
    /* Work on a copy to avoid modifying probs */
    for (int i = 0; i < k; i++) {
        int best = -1;
        float best_val = -FLT_MAX;

        for (int j = 0; j < n; j++) {
            /* Skip already selected */
            int skip = 0;
            for (int s = 0; s < i; s++) {
                if (indices[s] == j) { skip = 1; break; }
            }
            if (skip) continue;

            if (probs[j] > best_val) {
                best_val = probs[j];
                best = j;
            }
        }

        indices[i] = best;
        values[i] = best_val;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * MoE Forward Pass
 * ═══════════════════════════════════════════════════════════════════════*/

void moe_forward(float *out, const float *input,
                 const MoELayer *moe, RunState *state,
                 const Config *config)
{
    int dim         = (int)config->dim;
    int n_experts   = (int)config->n_experts;
    int n_active    = (int)config->n_active_experts;
    int expert_dim  = (int)config->expert_dim;

    /* 1. Compute gating logits: input[1×dim] × gate[dim×n_experts] */
    tensor_matmul(state->gate_logits, input, &moe->gate, 1, dim, n_experts);

    /* 2. Softmax */
    softmax(state->gate_logits, n_experts);

    /* 3. Top-K selection */
    int   top_indices[8];  /* max 8 active experts */
    float top_values[8];
    topk(state->gate_logits, n_experts, n_active, top_indices, top_values);

    /* Normalize selected weights */
    float weight_sum = 0.0f;
    for (int i = 0; i < n_active; i++) {
        weight_sum += top_values[i];
    }
    if (weight_sum > 0.0f) {
        for (int i = 0; i < n_active; i++) {
            top_values[i] /= weight_sum;
        }
    }

    /* 4. Zero output */
    vec_zero(out, dim);

    /* 5. Compute each selected expert and combine */
    for (int i = 0; i < n_active; i++) {
        int e = top_indices[i];
        float w = top_values[i];
        const ExpertMLP *expert = &moe->experts[e];

        /* SwiGLU: out = W2(SiLU(W1·x) * W3·x) */

        /* hb = W1·input  [expert_dim] */
        tensor_matmul(state->hb, input, &expert->w1, 1, dim, expert_dim);

        /* hb2 = W3·input [expert_dim] */
        tensor_matmul(state->hb2, input, &expert->w3, 1, dim, expert_dim);

        /* hb = SiLU(hb) * hb2 */
        silu_inplace(state->hb, expert_dim);
        vec_mul(state->hb, state->hb, state->hb2, expert_dim);

        /* expert_out = W2·hb [dim] */
        tensor_matmul(state->expert_out, state->hb, &expert->w2,
                      1, expert_dim, dim);

        /* Weighted accumulation */
        for (int d = 0; d < dim; d++) {
            out[d] += w * state->expert_out[d];
        }
    }
}
