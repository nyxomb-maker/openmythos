/*
 * quantize.c — INT4 dequantization implementation
 */

#include "quantize.h"

void dequantize_int4(float *out, const uint8_t *packed,
                     float scale, int count)
{
    int n_bytes = (count + 1) / 2;

    for (int b = 0; b < n_bytes; b++) {
        uint8_t byte = packed[b];

        /* Low nibble → even index */
        int idx0 = b * 2;
        if (idx0 < count) {
            int val0 = (int)(byte & 0x0F) - 8;
            out[idx0] = (float)val0 * scale;
        }

        /* High nibble → odd index */
        int idx1 = b * 2 + 1;
        if (idx1 < count) {
            int val1 = (int)(byte >> 4) - 8;
            out[idx1] = (float)val1 * scale;
        }
    }
}
