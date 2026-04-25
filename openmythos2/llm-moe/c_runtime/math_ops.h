/*
 * math_ops.h — Core mathematical operations for the LLM runtime
 *
 * Provides: matmul, rmsnorm, softmax, rope, silu, gelu
 * All operations work on flat float arrays for cache efficiency.
 */

#ifndef MATH_OPS_H
#define MATH_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Matrix Multiplication ──────────────────────────────────────────────
 * C[M×N] = A[M×K] × B[K×N]
 * Tiled for cache efficiency.
 */
void matmul(float *out, const float *a, const float *b,
            int M, int K, int N);

/* ── INT4 Matrix Multiplication ─────────────────────────────────────────
 * C[M×N] = A[M×K] × dequant(B_int4[K×N], scale)
 * B is packed: 2 INT4 values per byte, biased by +8.
 */
void matmul_int4(float *out, const float *a, const uint8_t *b_packed,
                 float scale, int M, int K, int N);

/* ── RMS Normalization ──────────────────────────────────────────────────
 * out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
 */
void rmsnorm(float *out, const float *x, const float *weight,
             int size, float eps);

/* ── Softmax (numerically stable) ───────────────────────────────────────
 * out[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
 */
void softmax(float *x, int size);

/* ── RoPE (Rotary Positional Embeddings) ────────────────────────────────
 * Apply rotary position encoding to q and k vectors.
 * dim = head_dim, pos = token position, theta = base frequency
 */
void rope(float *q, float *k, int dim, int pos, float theta);

/* ── Activation Functions ───────────────────────────────────────────────*/
void silu_inplace(float *x, int size);
void gelu_inplace(float *x, int size);

/* ── Element-wise Operations ────────────────────────────────────────────*/
void vec_add(float *out, const float *a, const float *b, int size);
void vec_mul(float *out, const float *a, const float *b, int size);
void vec_scale(float *out, const float *a, float scale, int size);
void vec_copy(float *dst, const float *src, int size);
void vec_zero(float *x, int size);

/* ── Reduction ──────────────────────────────────────────────────────────*/
float vec_dot(const float *a, const float *b, int size);
float vec_max(const float *x, int size);
float vec_sum(const float *x, int size);

#ifdef __cplusplus
}
#endif

#endif /* MATH_OPS_H */
