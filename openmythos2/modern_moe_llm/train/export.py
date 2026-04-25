import json
import os
import torch
import numpy as np
try:
    import gguf
except ImportError:
    print("Mock Mode: gguf package is missing. Install with 'pip install gguf'. Using a stub.")
    class gguf:
        class GGUFWriter:
            def __init__(self, path, arch): self.path = path
            def add_architecture(self): pass
            def add_uint32(self, key, val): pass
            def add_bool(self, key, val): pass
            def add_tensor(self, name, data): pass
            def write_header_to_file(self): pass
            def write_kv_data_to_file(self): pass
            def write_tensors_to_file(self): pass
            def close(self): pass

from model import ModernMoELLM, ModelConfig

TOKENIZER_DIR = "/home/miguelbds/Área de trabalho/Nova pasta/Models/"

def convert_to_gguf(model: ModernMoELLM, output_path: str):
    gguf_writer = gguf.GGUFWriter(output_path, "modern_moe")
    
    # 1. Configurações da Arquitetura
    gguf_writer.add_architecture()
    gguf_writer.add_uint32("modern_moe.vocab_size", model.config.vocab_size)
    gguf_writer.add_uint32("modern_moe.context_length", model.config.max_position_embeddings)
    gguf_writer.add_uint32("modern_moe.embedding_length", model.config.hidden_size)
    gguf_writer.add_uint32("modern_moe.block_count", model.config.num_hidden_layers)
    gguf_writer.add_uint32("modern_moe.feed_forward_length", model.config.intermediate_size)
    gguf_writer.add_uint32("modern_moe.attention.head_count", model.config.num_attention_heads)
    gguf_writer.add_uint32("modern_moe.attention.head_count_kv", model.config.num_key_value_heads)
    
    # Configurações Específicas do Projeto
    gguf_writer.add_uint32("modern_moe.expert_count", model.config.num_experts)
    gguf_writer.add_uint32("modern_moe.expert_used_count", model.config.num_experts_per_tok)
    gguf_writer.add_uint32("modern_moe.compression_dim", model.config.compression_dim)
    
    # Indica explicitamente no binário que o C Runtime deve criar o KV em 2-bit turbo quant
    gguf_writer.add_bool("modern_moe.kv_cache.turbo_quant_2bit", True)
    
    # 2. Leitura do Tokenizer Local
    try:
        tok_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
        if os.path.exists(tok_path):
            with open(tok_path, "r", encoding="utf-8") as f:
                tok_data = json.load(f)
            vocab = tok_data.get("model", {}).get("vocab", {})
            print(f"Info: Tokenizer detectado em {TOKENIZER_DIR} com {len(vocab)} palavras no dicionário (BPE structure).")
            # Em uma implementação completa da bilbioteca gguf python, preencheríamos os arrays de strings e byte-types.
    except Exception as e:
        print(f"Aviso: Não foi possível processar o tokenizer local: {e}")

    # 3. Serialização dos Tensores rigorosamente em FP16
    print("Formatando tensores em FP16 e alocando na RAM...")
    state_dict = model.state_dict()
    for name, tensor in state_dict.items():
        tensor_fp16 = tensor.to(torch.float16).contiguous().numpy()
        gguf_writer.add_tensor(name, tensor_fp16)
        
    print("Escrevendo binário MMF para o disco...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"Sucesso! Modelo {output_path} exportado.")

if __name__ == "__main__":
    print("Inicializando Modelo em PyTorch [Modo Demonstração]...")
    # Em produção, carregaríamos um checkpoint aqui
    config = ModelConfig(num_hidden_layers=2) # Reduzido para testar na máquina local rapidamente
    model = ModernMoELLM(config)
    convert_to_gguf(model, "modern_moe_fp16.gguf")
