/*
 * math_ops.c — Core mathematical operations implementation
 *
 * Optimizations:
 *   - Tiled matmul for L1/L2 cache efficiency
 *   - SIMD-friendly memory access patterns
 *   - OpenMP parallelization for large operations
 *   - Numerically stable softmax
 */

#include "math_ops.h"
#include <math.h>
#include <float.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Tile size for cache-blocked matmul */
#define TILE_SIZE 32

/* ═══════════════════════════════════════════════════════════════════════
 * Matrix Multiplication (tiled)
 * C[M×N] = A[M×K] × B[K×N]
 * ═══════════════════════════════════════════════════════════════════════*/
void matmul(float *out, const float *a, const float *b,
            int M, int K, int N)
{
    /* Zero output */
    memset(out, 0, (size_t)M * N * sizeof(float));

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i0 = 0; i0 < M; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {
            for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                /* Tile bounds */
                int i_end = (i0 + TILE_SIZE < M) ? i0 + TILE_SIZE : M;
                int j_end = (j0 + TILE_SIZE < N) ? j0 + TILE_SIZE : N;
                int k_end = (k0 + TILE_SIZE < K) ? k0 + TILE_SIZE : K;

                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a_ik = a[i * K + k];
                        for (int j = j0; j < j_end; j++) {
                            out[i * N + j] += a_ik * b[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * INT4 Matrix Multiplication with on-the-fly dequantization
 * C[M×N] = A[M×K] × dequant(B_packed[K×N])
 *
 * B is packed: 2 INT4 values per byte, biased by +8.
 * byte = (val1_biased & 0x0F) | (val2_biased << 4)
 * Actual value = (biased - 8) * scale
 * ═══════════════════════════════════════════════════════════════════════*/
void matmul_int4(float *out, const float *a, const uint8_t *b_packed,
                 float scale, int M, int K, int N)
{
    memset(out, 0, (size_t)M * N * sizeof(float));

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = a[i * K + k];
            for (int j = 0; j < N; j++) {
                /* Linear index in the K×N matrix */
                int idx = k * N + j;
                int byte_idx = idx / 2;
                uint8_t packed = b_packed[byte_idx];

                float val;
                if (idx % 2 == 0) {
                    /* Low nibble */
                    val = (float)((int)(packed & 0x0F) - 8) * scale;
                } else {
                    /* High nibble */
                    val = (float)((int)(packed >> 4) - 8) * scale;
                }

                out[i * N + j] += a_ik * val;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * RMS Normalization
 * out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
 * ═══════════════════════════════════════════════════════════════════════*/
void rmsnorm(float *out, const float *x, const float *weight,
             int size, float eps)
{
    /* Compute mean of squares */
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = 1.0f / sqrtf(sum_sq / (float)size + eps);

    /* Normalize and scale */
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * rms * weight[i];
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Softmax (numerically stable)
 * In-place: x[i] = exp(x[i] - max) / sum(exp)
 * ═══════════════════════════════════════════════════════════════════════*/
void softmax(float *x, int size)
{
    /* Find max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * RoPE (Rotary Positional Embeddings)
 *
 * For each pair (i, i+1) of dimensions:
 *   freq = pos / (theta ^ (2*i/dim))
 *   (q[i], q[i+1]) = (q[i]*cos - q[i+1]*sin, q[i]*sin + q[i+1]*cos)
 *   Same for k.
 * ═══════════════════════════════════════════════════════════════════════*/
void rope(float *q, float *k, int dim, int pos, float theta)
{
    for (int i = 0; i < dim; i += 2) {
        float freq = 1.0f / powf(theta, (float)i / (float)dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        /* Rotate query */
        float q0 = q[i];
        float q1 = q[i + 1];
        q[i]     = q0 * cos_a - q1 * sin_a;
        q[i + 1] = q0 * sin_a + q1 * cos_a;

        /* Rotate key */
        float k0 = k[i];
        float k1 = k[i + 1];
        k[i]     = k0 * cos_a - k1 * sin_a;
        k[i + 1] = k0 * sin_a + k1 * cos_a;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Activation Functions
 * ═══════════════════════════════════════════════════════════════════════*/

void silu_inplace(float *x, int size)
{
    for (int i = 0; i < size; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));  /* x * sigmoid(x) */
    }
}

void gelu_inplace(float *x, int size)
{
    /* GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    const float c = 0.7978845608f; /* sqrt(2/pi) */
    for (int i = 0; i < size; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Element-wise Operations
 * ═══════════════════════════════════════════════════════════════════════*/

void vec_add(float *out, const float *a, const float *b, int size)
{
    for (int i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}

void vec_mul(float *out, const float *a, const float *b, int size)
{
    for (int i = 0; i < size; i++) {
        out[i] = a[i] * b[i];
    }
}

void vec_scale(float *out, const float *a, float scale, int size)
{
    for (int i = 0; i < size; i++) {
        out[i] = a[i] * scale;
    }
}

void vec_copy(float *dst, const float *src, int size)
{
    memcpy(dst, src, (size_t)size * sizeof(float));
}

void vec_zero(float *x, int size)
{
    memset(x, 0, (size_t)size * sizeof(float));
}

/* ═══════════════════════════════════════════════════════════════════════
 * Reduction Operations
 * ═══════════════════════════════════════════════════════════════════════*/

float vec_dot(const float *a, const float *b, int size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

float vec_max(const float *x, int size)
{
    float m = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > m) m = x[i];
    }
    return m;
}

float vec_sum(const float *x, int size)
{
    float s = 0.0f;
    for (int i = 0; i < size; i++) {
        s += x[i];
    }
    return s;
}
