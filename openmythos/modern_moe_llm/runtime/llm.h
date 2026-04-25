#ifndef LLM_H
#define LLM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Definindo os tipos de dados suportados pelo nosso Custom C Runtime
typedef enum {
    TYPE_F32,
    TYPE_F16,
    TYPE_Q2_K_TURBO // Turbo Quant 2-bit (Usado para o KV Cache)
} TensorType;

typedef struct {
    int dim0, dim1, dim2;
    TensorType type;
    void *data;
} Tensor;

// Configuração principal da arquitetura
typedef struct {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int max_position_embeddings;
    int num_experts;
    int num_experts_per_tok;
    int compression_dim;
    float rms_norm_eps;
} ModelConfig;

// Estruturas de Pesos do MoE
typedef struct {
    Tensor *gate_weight; // [num_experts, hidden_size]
    Tensor **w1; // Array de num_experts tensores
    Tensor **w2;
    Tensor **w3;
} MoELayer;

// Estrutura de Attention com Resíduos
typedef struct {
    Tensor *q_proj;
    Tensor *k_proj;
    Tensor *v_proj;
    Tensor *o_proj;
    // Resíduo de atenção
    Tensor *mix_proj;
} Attention;

// Bloco completo do Transformer
typedef struct {
    Tensor *input_layernorm;
    Tensor *post_attention_layernorm;
    Tensor *per_layer_compress; // Para embeddings de memória SSD
    Attention attn;
    MoELayer moe;
} TransformerLayer;

// KV Cache entry para paginação
typedef struct {
    int token_index;
    Tensor *k_cache; // Alocado em TYPE_Q2_K_TURBO
    Tensor *v_cache; // Alocado em TYPE_Q2_K_TURBO
    Tensor *layer_embedding; // Compressão da camada (ram -> ssd)
    bool is_in_ssd;  
} KVCachePage;

// Estrutura raiz do Modelo C
typedef struct {
    ModelConfig config;
    Tensor *token_embedding;
    TransformerLayer *layers;
    Tensor *final_norm;
    Tensor *lm_head;
    
    // Gerenciador de contexto
    KVCachePage **kv_pages; 
    int seq_len;
} Model;

// API do Model.c e Tensor.c
Tensor* tensor_alloc(int dim0, int dim1, int dim2, TensorType type);
void tensor_free(Tensor *t);
// Forward pass: Recebe um token_id e o KVCache local, retorna a próxima predição
float* model_forward(Model *model, int token_id, int pos);
Model* load_gguf_model(const char* filepath);

#endif // LLM_H
