"""
Transformer Decoder-only with Mixture of Experts
=================================================
Full PyTorch implementation including:
- RMSNorm
- Rotary Positional Embeddings (RoPE)
- Causal Multi-Head Self-Attention with KV Cache
- Mixture of Experts (MoE) with Top-K gating and load balancing
- Weight Tying (embedding ↔ output)
- Gradient Checkpointing support
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from config import ModelConfig


# ═══════════════════════════════════════════════════════════════════════════
# RMSNorm
# ═══════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Simpler and faster than LayerNorm — no mean centering.
    output = x * weight / sqrt(mean(x^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


# ═══════════════════════════════════════════════════════════════════════════
# Rotary Positional Embeddings (RoPE)
# ═══════════════════════════════════════════════════════════════════════════

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex exponentials for RoPE.

    Returns:
        freqs_cis: [max_seq_len, dim//2] complex tensor
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)  # [max_seq_len, dim//2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.

    Args:
        xq: [batch, seq_len, n_heads, head_dim]
        xk: [batch, seq_len, n_heads, head_dim]
        freqs_cis: [seq_len, head_dim//2] complex

    Returns:
        Rotated (xq, xk)
    """
    # Reshape to complex: [..., head_dim] → [..., head_dim//2, 2] → complex
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis: [seq_len, head_dim//2] → [1, seq_len, 1, head_dim//2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    xq_out = torch.view_as_real(xq_c * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ═══════════════════════════════════════════════════════════════════════════
# Causal Multi-Head Self-Attention
# ═══════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE and KV cache support.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dim = config.dim

        # QKV projections (fused for efficiency)
        self.wq = nn.Linear(config.dim, config.dim, bias=False)
        self.wk = nn.Linear(config.dim, config.dim, bias=False)
        self.wv = nn.Linear(config.dim, config.dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, dim]
            freqs_cis: [seq_len, head_dim//2] complex
            mask: [seq_len, seq_len] or None
            kv_cache: (cached_k, cached_v) or None
            start_pos: position offset for KV cache

        Returns:
            output: [batch, seq_len, dim]
            new_kv_cache: (keys, values)
        """
        B, T, D = x.shape

        # Project
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rope(q, k, freqs_cis)

        # KV Cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        new_kv_cache = (k, v)

        # Transpose for attention: [B, n_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, n_heads, T_q, T_kv]

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)

        # Weighted sum
        out = torch.matmul(attn, v)  # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.wo(out), new_kv_cache


# ═══════════════════════════════════════════════════════════════════════════
# Expert MLP (SwiGLU)
# ═══════════════════════════════════════════════════════════════════════════

class ExpertMLP(nn.Module):
    """
    Single expert feedforward network using SwiGLU activation.
    SwiGLU(x) = SiLU(xW1) * (xW2)
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ═══════════════════════════════════════════════════════════════════════════
# Mixture of Experts
# ═══════════════════════════════════════════════════════════════════════════

class MoELayer(nn.Module):
    """
    Mixture of Experts layer with Top-K gating and load balancing loss.

    Architecture:
        1. Gating: Linear(dim, n_experts) → Softmax → Top-K
        2. Dispatch: route each token to selected experts
        3. Combine: weighted sum of expert outputs
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_active = config.n_active_experts

        # Gate network
        self.gate = nn.Linear(config.dim, config.n_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList([
            ExpertMLP(config.dim, config.expert_dim)
            for _ in range(config.n_experts)
        ])

        # Auxiliary loss accumulator
        self.aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]

        Returns:
            output: [batch, seq_len, dim]
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [B*T, D]
        N = x_flat.shape[0]

        # ── Gating ──
        gate_logits = self.gate(x_flat)               # [N, n_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)    # [N, n_experts]

        # Top-K selection
        topk_vals, topk_idx = torch.topk(gate_probs, self.n_active, dim=-1)  # [N, K]

        # Normalize selected weights
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # ── Load Balancing Loss ──
        # Fraction of tokens routed to each expert
        # f_i = (1/N) * sum_x I(expert_i in top-k for x)
        one_hot = F.one_hot(topk_idx, self.n_experts).float()  # [N, K, n_experts]
        tokens_per_expert = one_hot.sum(dim=1).mean(dim=0)      # [n_experts]

        # Average gate probability per expert
        avg_gate_prob = gate_probs.mean(dim=0)  # [n_experts]

        # Auxiliary loss: n_experts * sum(f_i * p_i) — encourages uniform distribution
        self.aux_loss = self.n_experts * (tokens_per_expert * avg_gate_prob).sum()

        # ── Dispatch & Combine ──
        output = torch.zeros_like(x_flat)

        for k in range(self.n_active):
            expert_indices = topk_idx[:, k]     # [N]
            expert_weights = topk_weights[:, k]  # [N]

            for e in range(self.n_experts):
                mask = (expert_indices == e)
                if not mask.any():
                    continue
                expert_input = x_flat[mask]
                expert_output = self.experts[e](expert_input)
                output[mask] += expert_weights[mask].unsqueeze(-1) * expert_output

        return output.view(B, T, D)


# ═══════════════════════════════════════════════════════════════════════════
# Transformer Block
# ═══════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Single transformer block: RMSNorm → Attention → RMSNorm → MoE
    Pre-norm architecture with residual connections.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.moe = MoELayer(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, dim]

        Returns:
            output: [batch, seq_len, dim]
            kv_cache: updated KV cache
        """
        # Attention with residual
        h = self.attn_norm(x)
        attn_out, new_kv_cache = self.attn(h, freqs_cis, mask, kv_cache, start_pos)
        x = x + attn_out

        # MoE with residual
        x = x + self.moe(self.ffn_norm(x))

        return x, new_kv_cache

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        return self.moe.aux_loss


# ═══════════════════════════════════════════════════════════════════════════
# Full Transformer LM
# ═══════════════════════════════════════════════════════════════════════════

class TransformerLM(nn.Module):
    """
    Full Transformer Language Model with MoE.

    Architecture:
        Token Embedding → [TransformerBlock × N] → RMSNorm → Linear (tied)

    Features:
        - Weight tying between embedding and output projection
        - RoPE (computed on-the-fly)
        - KV cache for efficient generation
        - Gradient checkpointing support
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.dim, config.norm_eps)

        # Output projection (weight-tied with embedding)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.embedding.weight  # Weight tying

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_rope_freqs(
            config.head_dim, config.max_seq_len, config.rope_theta
        )

        # Gradient checkpointing flag
        self.use_checkpointing = False

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_checkpointing = True

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Forward pass.

        Args:
            tokens: [batch, seq_len] token IDs
            start_pos: position offset (for KV cache generation)
            kv_caches: list of (k, v) caches per layer, or None

        Returns:
            logits: [batch, seq_len, vocab_size]
            new_kv_caches: updated KV caches
            aux_loss: total MoE load balancing loss
        """
        B, T = tokens.shape

        # Embedding
        h = self.embedding(tokens)  # [B, T, dim]

        # RoPE frequencies for this sequence
        device = tokens.device
        freqs_cis = self.freqs_cis[start_pos: start_pos + T].to(device)

        # Causal mask
        if T > 1:
            mask = torch.full((T, T), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            if kv_caches is not None and kv_caches[0] is not None:
                # Extend mask to cover cached positions
                cache_len = kv_caches[0][0].shape[1]
                mask = torch.cat([
                    torch.zeros(T, cache_len, device=device),
                    mask,
                ], dim=1)
        else:
            mask = None

        # Transformer layers
        new_kv_caches = []
        total_aux_loss = torch.tensor(0.0, device=device)

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None

            if self.use_checkpointing and self.training:
                # Gradient checkpointing — recompute during backward pass
                def create_custom_forward(module):
                    def custom_forward(*args):
                        out, kv = module(*args)
                        return out, kv[0], kv[1]
                    return custom_forward

                h, cached_k, cached_v = gradient_checkpoint(
                    create_custom_forward(layer),
                    h, freqs_cis, mask, kv_cache, start_pos,
                    use_reentrant=False,
                )
                new_kv = (cached_k, cached_v)
            else:
                h, new_kv = layer(h, freqs_cis, mask, kv_cache, start_pos)

            new_kv_caches.append(new_kv)

            if layer.aux_loss is not None:
                total_aux_loss = total_aux_loss + layer.aux_loss

        # Final norm + output projection
        h = self.norm(h)
        logits = self.output(h)

        return logits, new_kv_caches, total_aux_loss

    @torch.inference_mode()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_id: int = 3,
    ) -> List[int]:
        """
        Autoregressive text generation with KV cache.

        Args:
            tokens: [1, seq_len] initial prompt token IDs
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus (top-p) filtering
            eos_id: end-of-sequence token ID

        Returns:
            List of generated token IDs (including prompt)
        """
        self.eval()
        generated = tokens.tolist()[0]
        kv_caches = None

        # Prefill: process entire prompt at once
        logits, kv_caches, _ = self.forward(tokens, start_pos=0, kv_caches=None)
        next_logits = logits[:, -1, :]  # [1, vocab]
        start_pos = tokens.shape[1]

        for _ in range(max_new_tokens):
            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                    next_logits[next_logits < v[:, [-1]]] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[sorted_mask] = float("-inf")
                    next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            next_id = next_token.item()
            generated.append(next_id)

            if next_id == eos_id:
                break

            # Decode step: single token with KV cache
            logits, kv_caches, _ = self.forward(
                next_token, start_pos=start_pos, kv_caches=kv_caches
            )
            next_logits = logits[:, -1, :]
            start_pos += 1

        return generated

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_active_parameters(self) -> int:
        """
        Count parameters active per forward pass.
        With MoE, only top-k experts are used per token.
        """
        total = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                if "experts" in name:
                    # Only n_active_experts / n_experts are used
                    total += p.numel() * self.config.n_active_experts / self.config.n_experts
                else:
                    total += p.numel()
        return int(total)
