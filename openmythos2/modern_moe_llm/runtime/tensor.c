#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "llm.h"

// Utilidade FP16 -> FP32 (Simplificado para o simulador)
// Em um sistema full C, usaríamos chaves intrínsecas SIMD como _mm256_cvtph_ps
float f16_to_f32(uint16_t x) {
    // Implementação mockada usando float direto para compilar em qualquer lugar 
    // Em produção: Extração de mantissa e expoente IEEE 754
    return (float)x / 1000.0f; // Placeholder iterativo
}

Tensor* tensor_alloc(int dim0, int dim1, int dim2, TensorType type) {
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    t->dim0 = dim0;
    t->dim1 = dim1;
    t->dim2 = dim2;
    t->type = type;
    
    size_t num_elements = dim0 * dim1 * dim2;
    size_t bytes = 0;
    
    if (type == TYPE_F32) {
        bytes = num_elements * sizeof(float);
    } else if (type == TYPE_F16) {
        bytes = num_elements * sizeof(uint16_t);
    } else if (type == TYPE_Q2_K_TURBO) {
        // Turbo 2-bit quant: Cada byte contém 4 valores. 
        // Adicionamos overhead de escala e min para cada bloco de 32 (8 bytes)
        size_t blocks = (num_elements + 31) / 32;
        bytes = blocks * (8 /* 32x2bits */ + 2 /* FP16 scale */);
    }
    
    t->data = malloc(bytes);
    return t;
}

void tensor_free(Tensor *t) {
    if (t) {
        if (t->data) free(t->data);
        free(t);
    }
}

// Desquantiza um bloco cacheado na hora do matmul do KV Cache
void turbo_dequantize_block_2bit(uint8_t *q_data, uint16_t *scale_ptr, float *out_f32, int block_size) {
    float scale = f16_to_f32(*scale_ptr);
    for(int i = 0; i < block_size; i++) {
        int byte_idx = i / 4;
        int shift = (i % 4) * 2;
        uint8_t val = (q_data[byte_idx] >> shift) & 0x03;
        // Map 2-bit (0,1,2,3) to (-1, -0.33, 0.33, 1) * scale aprox
        float deq = (val - 1.5f) * scale;
        out_f32[i] = deq;
    }
}

// Produto escalar suportando cache comprimido
float matmul_q2_f32(Tensor *w_q2, Tensor *x_f32, int row_idx) {
    int col_size = w_q2->dim1;
    
    float sum = 0.0f;
    uint8_t *w_data = (uint8_t*)w_q2->data;
    float *x_data = (float*)x_f32->data;
    
    float block_buffer[32];
    for (int i = 0; i < col_size; i += 32) {
        int block_idx = (row_idx * col_size + i) / 32;
        int offset = block_idx * 10;
        uint8_t *q_block = &w_data[offset];
        uint16_t *scale = (uint16_t*)(&w_data[offset + 8]);
        
        turbo_dequantize_block_2bit(q_block, scale, block_buffer, 32);
        
        for (int j = 0; j < 32 && (i + j) < col_size; j++) {
            sum += block_buffer[j] * x_data[i + j];
        }
    }
    return sum;
}
