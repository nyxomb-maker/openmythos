/*
 * quantize.h — INT4 dequantization utilities
 */

#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dequantize INT4 packed data to FP32 buffer.
 * packed: byte array with 2 INT4 values per byte (biased +8)
 * out:    FP32 output buffer (must hold `count` floats)
 * scale:  per-tensor scale factor
 * count:  number of float values to unpack
 */
void dequantize_int4(float *out, const uint8_t *packed,
                     float scale, int count);

/* Dequantize a single INT4 value at a given index. */
static inline float dequant_int4_single(const uint8_t *packed,
                                        float scale, int idx)
{
    uint8_t byte = packed[idx / 2];
    int val;
    if (idx % 2 == 0) {
        val = (int)(byte & 0x0F) - 8;
    } else {
        val = (int)(byte >> 4) - 8;
    }
    return (float)val * scale;
}

#ifdef __cplusplus
}
#endif

#endif /* QUANTIZE_H */
