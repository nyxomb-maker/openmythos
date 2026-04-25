#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llm.h"

#define MAX_RAM_TOKENS 65536 // 64K
#define MAX_SSD_TOKENS 1000000 // 1M

// Sistema mock de mmap para o SSD KV cache
typedef struct {
    char filepath[256];
    FILE *fd;
    int current_ssd_tokens;
} SSDManager;

SSDManager global_ssd;

void memory_manager_init() {
    strcpy(global_ssd.filepath, "kv_cache.ssd.bin");
    global_ssd.fd = fopen(global_ssd.filepath, "wb+");
    global_ssd.current_ssd_tokens = 0;
    printf("Memory Manager initialized: up to %d tokens in RAM, %d in SSD.\n", MAX_RAM_TOKENS, MAX_SSD_TOKENS);
}

// Quando o contexto cresce além da RAM, usamos a Similaridade Cosseno no PerLayerEmbedding
// para decidir qual bloco enviar para o SSD.
float cosine_similarity(Tensor *a, Tensor *b) {
    // Calculo mockado, em prod seria sum(a[i]*b[i]) / (norm(a)*norm(b))
    return 1.0f; 
}

// Faz downgrade de um token menos relevante da RAM para o SSD
void swap_page_to_ssd(KVCachePage *page) {
    if (global_ssd.current_ssd_tokens >= MAX_SSD_TOKENS) {
        printf("WARNING: SSD KV Cache cheio!\n");
        return;
    }
    
    // Escreve os bytes Q2_K_TURBO direto pro binário
    // (mockado) fwrite(page->k_cache->data, 1, byte_size, global_ssd.fd);
    
    // Libera a ram
    tensor_free(page->k_cache);
    tensor_free(page->v_cache);
    
    page->k_cache = NULL;
    page->v_cache = NULL;
    page->is_in_ssd = true;
    
    global_ssd.current_ssd_tokens++;
}

// Carrega de volta para RAM
void swap_page_to_ram(KVCachePage *page, ModelConfig *cfg) {
    if (!page->is_in_ssd) return;
    
    // Aloca usando TYPE_Q2_K_TURBO
    page->k_cache = tensor_alloc(cfg->num_key_value_heads, cfg->hidden_size/cfg->num_attention_heads, 1, TYPE_Q2_K_TURBO);
    page->v_cache = tensor_alloc(cfg->num_key_value_heads, cfg->hidden_size/cfg->num_attention_heads, 1, TYPE_Q2_K_TURBO);
    
    // fseek(..., page_offset); fread(...);
    
    page->is_in_ssd = false;
    global_ssd.current_ssd_tokens--;
}

// Analisa a janela atual e remove contexto que destoa do Per-Layer Embedding atual
void check_and_evict_memory(Model *model, int current_pos, Tensor *current_layer_embed) {
    if (current_pos < MAX_RAM_TOKENS) {
        return; // RAM suporta tranquilamente
    }
    
    // Caso de uso: Evictar as memórias menos relevantes
    // (exceção: manter prompt de sistema e últimos K tokens intactos)
    for (int i = 0; i < current_pos; i++) {
        KVCachePage *p = model->kv_pages[i];
        if (!p->is_in_ssd && p->layer_embedding) {
            float sim = cosine_similarity(p->layer_embedding, current_layer_embed);
            if (sim < 0.2f) { // Limiar de teste
                swap_page_to_ssd(p);
            }
        }
    }
}
