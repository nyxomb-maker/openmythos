import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig

class AttentionResidual(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Para resíduo de atenção, usamos uma projeção linear simples caso queiramos combinar
        # o estado de atenção anterior com o atual.
        self.mix_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, current_attn, prev_attn):
        if prev_attn is None:
            return current_attn
        # Retenção de fluxo (Kimi-like) combinando attn atual e anterior
        mixed = torch.cat([current_attn, prev_attn], dim=-1)
        return self.mix_proj(mixed)

class PerLayerEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Comprimir a saída de cada camada para a memória estendida
        self.compress = nn.Linear(config.hidden_size, config.compression_dim)

    def forward(self, hidden_states):
        # A compressão só precisa ocorrer em runtime de inferência para KV SSD.
        # No treinamento, isso serve apenas pra alinhar formas e prever perda se necessário.
        return self.compress(hidden_states)

class MoELayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        
        # Gating network
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        
        # Experts
        self.w1 = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size, config.intermediate_size)) for _ in range(self.num_experts)])
        self.w2 = nn.ParameterList([nn.Parameter(torch.empty(config.intermediate_size, config.hidden_size)) for _ in range(self.num_experts)])
        self.w3 = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size, config.intermediate_size)) for _ in range(self.num_experts)])
        
        for i in range(self.num_experts):
            nn.init.normal_(self.w1[i], std=0.02)
            nn.init.normal_(self.w2[i], std=0.02)
            nn.init.normal_(self.w3[i], std=0.02)

    def forward(self, hidden_states):
        batch_size, seq_len, d_model = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, d_model)
        
        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        
        # Sparse dispatch simplificado para legibilidade (em um framework real usaríamos scatter/gather otimizado)
        for i in range(self.num_experts):
            expert_mask = (selected_experts == i).any(dim=-1)
            if not expert_mask.any():
                continue
                
            expert_indices = torch.nonzero(expert_mask).squeeze(-1)
            tokens_for_expert = hidden_states_flat[expert_indices]
            
            w1_expert = self.w1[i]
            w2_expert = self.w2[i]
            w3_expert = self.w3[i]
            
            # SwiGLU 
            act = F.silu(tokens_for_expert @ w1_expert) * (tokens_for_expert @ w3_expert)
            out = act @ w2_expert
            
            # Ponderação
            token_weights = routing_weights[expert_mask]
            expert_col_idx = (selected_experts[expert_mask] == i).nonzero(as_tuple=True)[1]
            token_weights = token_weights[torch.arange(len(token_weights)), expert_col_idx].unsqueeze(-1)
            
            final_hidden_states[expert_indices] += out * token_weights

        return final_hidden_states.view(batch_size, seq_len, d_model)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# Uma atenção simplificada referenciando o RoPE e KV cache
class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, position_ids, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()
        
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # RoPE application omitido para brevidade (será focado no C)
        
        # GQA repeat interleave
        k = torch.repeat_interleave(k, dim=1, repeats=self.num_key_value_groups)
        v = torch.repeat_interleave(v, dim=1, repeats=self.num_key_value_groups)
        
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.attn_residual = AttentionResidual(config)
        self.per_layer_embed = PerLayerEmbedding(config)
        self.moe = MoELayer(config)

    def forward(self, hidden_states, position_ids, prev_attn=None):
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        current_attn = self.self_attn(hidden_states, position_ids)
        
        # Attention Residual mix
        mixed_attn = self.attn_residual(current_attn, prev_attn)
        hidden_states = residual + mixed_attn
        
        # MoE com Feed Forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        
        # Embeddings por camada exportados
        layer_compressed_embed = self.per_layer_embed(hidden_states)
        
        return hidden_states, mixed_attn, layer_compressed_embed

class ModernMoELLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            
        prev_attn = None
        layer_embeds = []
        
        for layer in self.layers:
            hidden_states, prev_attn, layer_emb = layer(hidden_states, position_ids, prev_attn)
            layer_embeds.append(layer_emb)
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, layer_embeds
