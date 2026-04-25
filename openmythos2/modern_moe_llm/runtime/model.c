#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llm.h"

// Função Stub para carregar do GGUF 
// (Na pŕatica leria os metadados mágicos 0x46554747 e mapearia os tensores FP16)
Model* load_gguf_model(const char* filepath) {
    printf("Loading GGUF Model from %s in FP16 mode...\n", filepath);
    
    Model *m = (Model*)malloc(sizeof(Model));
    // Inicialização da config (Baseado no PyTorch)
    m->config.vocab_size = 32000;
    m->config.hidden_size = 4096;
    m->config.num_hidden_layers = 32;
    m->config.num_experts = 64;
    m->config.num_experts_per_tok = 8;
    
    m->token_embedding = tensor_alloc(m->config.vocab_size, m->config.hidden_size, 1, TYPE_F16);
    m->layers = (TransformerLayer*)malloc(sizeof(TransformerLayer) * m->config.num_hidden_layers);
    
    for (int i=0; i<m->config.num_hidden_layers; i++) {
        // Alocar MoE Gating
        m->layers[i].moe.gate_weight = tensor_alloc(m->config.num_experts, m->config.hidden_size, 1, TYPE_F16);
    }
    printf("MoE layers allocated (64 experts per layer, top-8 sparse routing active).\n");
    return m;
}

// MoE forward: calcula roteamento
void moe_forward(MoELayer *moe, ModelConfig *cfg, Tensor *hidden_state, Tensor *out) {
    // 1. Calcular predições do Gate: logits = hidden_state @ gate_weight^T
    // 2. Softmax
    // 3. Selecionar top_k (8 experts)
    // 4. sparse distpach e acumular no Tensor *out
    
    // Simulação do roteamento apenas pra demonstração da estrutura
    // printf("Routing token to 8 of %d experts...\n", cfg->num_experts);
}

// Pipeline de uma camada com Residuos e Compressão
void layer_forward(Model *model, int layer_idx, Tensor *hidden_state, Tensor *prev_attn, KVCachePage *kv_cache) {
    TransformerLayer *l = &model->layers[layer_idx];
    
    // 1. RMSNorm
    // 2. Attention (QKV)
    // Se o KV-Cache não está na RAM e layer precisa, fazer swap do SSD!
    if (kv_cache->is_in_ssd) {
        printf("Swapping KV cache from SSD for token %d (Layer %d)\n", kv_cache->token_index, layer_idx);
        // Load Q2_K_TURBO from SSD
    }
    
    // Attention Residual (Kimi-like)
    // Se prev_attn existe, combina com a current att:
    // mix = linear(cat(current_attn, prev_attn))
    
    // 3. MoE Feed Forward
    moe_forward(&l->moe, &model->config, hidden_state, hidden_state);
    
    // 4. Per_layer Embedding Extractor
    // Extract `compression_dim` feature based on the hiddenstate and store in the KVCachePage
    // kv_cache->layer_embedding = linear_compress(hidden_state);
}

float* model_forward(Model *model, int token_id, int pos) {
    // Pegar token_embedding do model->token_embedding[token_id]
    Tensor *hidden_state = tensor_alloc(model->config.hidden_size, 1, 1, TYPE_F32);
    Tensor *prev_attn = NULL;
    
    for(int i=0; i < model->config.num_hidden_layers; i++) {
        layer_forward(model, i, hidden_state, prev_attn, model->kv_pages[pos]);
        // Update prev_attn here
    }
    
    // norm + final projection against lm_head pra retornar os logits
    float* logits = (float*)malloc(sizeof(float) * model->config.vocab_size);
    tensor_free(hidden_state);
    return logits;
}
