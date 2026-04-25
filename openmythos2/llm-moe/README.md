# LLM-MoE — Transformer Decoder-only com Mixture of Experts

Implementação completa de um LLM moderno com foco em eficiência, escalabilidade e raciocínio estruturado.

## Arquitetura

```
Input Tokens
     │
     ▼
┌─────────────┐
│  Embedding  │ (Weight Tying ↔ Output)
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│         Transformer Block × N        │
│                                      │
│  ┌─────────┐    ┌──────────────────┐ │
│  │ RMSNorm │───▶│ Causal Attention │ │
│  └─────────┘    │ + RoPE + KVCache │ │
│       │         └────────┬─────────┘ │
│       │    + residual    │           │
│       ◄──────────────────┘           │
│       │                              │
│  ┌─────────┐    ┌──────────────────┐ │
│  │ RMSNorm │───▶│  Mixture of      │ │
│  └─────────┘    │  Experts (MoE)   │ │
│       │         │  Top-K Gating    │ │
│       │         │  SwiGLU Experts  │ │
│       │         └────────┬─────────┘ │
│       │    + residual    │           │
│       ◄──────────────────┘           │
└──────────────────┬───────────────────┘
                   │
              ┌────┴────┐
              │ RMSNorm │
              └────┬────┘
                   │
              ┌────┴────┐
              │ Output  │ (tied with Embedding)
              └────┬────┘
                   │
                   ▼
              Logits [vocab_size]
```

## Estrutura do Projeto

```
llm-moe/
├── python/
│   ├── config.py       # Configurações e presets
│   ├── tokenizer.py    # BPE tokenizer
│   ├── model.py        # Transformer + MoE (PyTorch)
│   ├── train.py        # Pipeline de treinamento
│   └── export.py       # Exportação INT4 quantizada
├── c_runtime/
│   ├── Makefile
│   ├── main.c          # Entry point + geração
│   ├── model.h/c       # Structs + loading + forward
│   ├── attention.h/c   # Flash Attention + KV Cache
│   ├── moe.h/c         # Mixture of Experts
│   ├── math_ops.h/c    # MatMul, RMSNorm, Softmax, RoPE
│   ├── quantize.h/c    # INT4 dequantização
│   ├── tokenizer.h/c   # BPE em C
│   └── reasoning.h/c   # Raciocínio estruturado
└── data/
    └── sample.txt      # Dados de demonstração
```

## Uso Rápido

### 1. Treinar (Python)

```bash
cd python

# Treinar tokenizer + modelo (config demo)
python train.py --config demo

# Exportar para binário INT4
python export.py --checkpoint ../checkpoints/model_step5000.pt --output ../model.bin
```

### 2. Compilar Runtime (C)

```bash
cd c_runtime

# Release build
make

# Com OpenMP (paralelo)
make openmp
```

### 3. Inferência

```bash
# Geração simples
./llm-moe ../model.bin "The Transformer architecture"

# Modo interativo
./llm-moe ../model.bin --interactive

# Com raciocínio estruturado
./llm-moe ../model.bin --reasoning "Explain why the sky is blue"

# Parâmetros de sampling
./llm-moe ../model.bin --temperature 0.5 --top-k 40 --top-p 0.9 "Hello"
```

## Componentes Técnicos

### Treinamento (Python/PyTorch)
- **Mixed Precision**: FP16/BF16 via torch.amp
- **Gradient Checkpointing**: trade compute for memory
- **AdamW**: com warmup linear + cosine decay
- **Weight Tying**: embedding ↔ output compartilham pesos
- **Load Balancing Loss**: distribui tokens entre experts

### Quantização (Export)
- **INT4**: 4 bits por peso, 2 valores por byte
- **Escala por tensor**: `scale = max(abs(W)) / 7`
- **Packing**: `byte = (val1+8 & 0x0F) | ((val2+8) << 4)`
- **Compressão**: ~8x vs FP32

### Runtime C
- **Flash Attention**: online softmax, O(1) memória por head
- **KV Cache**: append-only, sem recomputação
- **RoPE**: rotação posicional aplicada a Q e K
- **RMSNorm**: normalização sem centering
- **SwiGLU**: `SiLU(W1·x) * W3·x` para cada expert
- **Sampling**: temperatura, top-k, top-p (nucleus)
- **Zero dependências**: apenas libc + libm

### Raciocínio Estruturado
- Tokens especiais: `<think>`, `<step>`, `<verify>`, `<conclude>`
- Decomposição automática em sub-etapas
- Validação heurística (repetição, diversidade, coerência)
- Composição de resultados parciais

## Configurações

| Parâmetro | Demo | Full |
|---|---|---|
| Camadas | 4 | 128 |
| Dimensão | 256 | 512 |
| Heads | 4 | 8 |
| Experts | 4 | 8 |
| Top-K | 2 | 2 |
| Vocab | 4096 | 32000 |
| Seq. Máx | 512 | 2048 |

## Otimizações Futuras

1. **GQA (Grouped-Query Attention)**: reduzir KV cache compartilhando heads
2. **Quantização per-channel**: melhor precisão que per-tensor
3. **Speculative Decoding**: draft model para acelerar geração
4. **SIMD/AVX**: instruções vetoriais no matmul
5. **mmap**: memory-mapped loading para reduzir startup time
6. **Paged KV Cache**: gerenciamento dinâmico de memória para contextos longos
7. **Sliding Window Attention**: complexidade linear para sequências longas
