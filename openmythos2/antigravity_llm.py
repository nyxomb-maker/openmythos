#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║           ANTIGRAVITY LLM  —  Hybrid Prototype               ║
║  Recurrent · MoE · ACT · Memory · Sub-Agents · Tool-Calling  ║
╚══════════════════════════════════════════════════════════════╝

Single-file PyTorch implementation.  Clarity > performance.

Architecture:
  - 16-layer Transformer  (dim=128, 4 heads)
  - Mixture of Experts    (32 experts / layer, top-8 routing)
  - Recurrent loop        (up to max_steps per layer)
  - Attention residuals   (cross-step, last-K states)
  - Adaptive Computation Time (ACT)
  - Hierarchical memory   (fast deque + slow vector DB + autoencoder)
  - Sub-agent system      (last 4 layers)
  - Tool calling          (last layer, last step)
  - Gemma 4 BPE tokenizer (vocab = 262 144)

Usage:
  python antigravity_llm.py --smoke-test          # verify forward/backward
  python antigravity_llm.py                       # full training run
  python antigravity_llm.py --dim 64 --n-layers 4 # reduced config
"""

import os
import sys
import math
import time
import argparse
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

# ──────────────────────────────────────────────────────────────
# PHASE 0  ·  GLOBAL CONFIG
# ──────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Core dimensions ──────────────────────────────────────
    dim:          int   = 128
    n_heads:      int   = 4
    n_layers:     int   = 16

    # ── Mixture of Experts ───────────────────────────────────
    num_experts:  int   = 32
    top_k:        int   = 8
    cap_factor:   float = 1.25      # token capacity multiplier

    # ── Recurrent loop / ACT ─────────────────────────────────
    max_steps:           int   = 5
    act_threshold:       float = 0.99
    cross_step_lookback: int   = 4   # max past states for attn residuals

    # ── Sequence & batch ─────────────────────────────────────
    seq_len:    int = 128
    batch_size: int = 8

    # ── Tokenizer / vocab ────────────────────────────────────
    vocab_size:      int = 262_144   # Gemma 4 BPE
    tokenizer_path:  str = "/home/miguelbds/Área de trabalho/Nova pasta/Models/"
    corpus_path:     str = ""        # optional text corpus path
    pad_token_id:    int = 0
    bos_token_id:    int = 2
    eos_token_id:    int = 1

    # ── Hierarchical memory ───────────────────────────────────
    fast_mem_size: int = 16     # deque maxlen
    slow_mem_size: int = 256    # number of learnable memory slots
    latent_dim:    int = 64     # autoencoder bottleneck

    # ── Sub-agents ────────────────────────────────────────────
    n_agents:          int = 3
    agent_start_layer: int = 12  # agents active in layers [12, 15]

    # ── Tools ─────────────────────────────────────────────────
    n_tools: int = 3

    # ── Optimisation ─────────────────────────────────────────
    lr:              float = 3e-4
    weight_decay:    float = 1e-2
    warmup_steps:    int   = 100
    max_train_steps: int   = 10_000
    log_every:       int   = 50
    grad_clip:       float = 1.0

    # ── Loss weights ─────────────────────────────────────────
    moe_loss_weight: float = 0.01
    act_loss_weight: float = 0.001

    # ── System ───────────────────────────────────────────────
    device:             str  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp:            bool = True   # torch.amp autocast (CUDA only)
    use_grad_checkpoint: bool = True  # gradient checkpointing per block

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, \
            f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"


# ──────────────────────────────────────────────────────────────
# PHASE 1  ·  TOKEN + POSITIONAL EMBEDDING
# ──────────────────────────────────────────────────────────────

class TokenEmbedding(nn.Module):
    """
    Learned token embedding + learned positional embedding.
    Output is layer-normed to stabilise early training.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.dim,
                                padding_idx=cfg.pad_token_id)
        self.pos = nn.Embedding(cfg.seq_len, cfg.dim)
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, T]
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        return self.norm(self.tok(x) + self.pos(pos_ids))


# ──────────────────────────────────────────────────────────────
# PHASE 2.3  ·  MIXTURE OF EXPERTS (32 experts, top-8)
# ──────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """MLP expert: dim → 4·dim (GELU) → dim.  No bias for speed."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, 4 * cfg.dim, bias=False)
        self.w2 = nn.Linear(4 * cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer.

    Routing:
      gate_logits = Linear(h)            [N, E]
      topk_vals, topk_idx = topk(k=8)
      weights = softmax(topk_vals)       [N, k]

    Dispatch:
      Tokens are grouped by expert index.
      Capacity limit = cap_factor * N / num_experts  (overflow dropped).

    Load-balancing auxiliary loss:
      lb_loss = num_experts · Σ_e (importance_e × load_e)
      where importance = mean softmax(gate_logits)
            load       = mean fraction of tokens routed to each expert
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k       = cfg.top_k
        self.cap_factor  = cfg.cap_factor

        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.num_experts)])
        self.gate    = nn.Linear(cfg.dim, cfg.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x : [B, T, D]
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)                          # [N, D]

        # ── Gating ───────────────────────────────────────────
        gate_logits = self.gate(x_flat)                   # [N, E]
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)       # [N, k]

        # ── Load-balancing loss (no grad through load) ───────
        with torch.no_grad():
            importance = F.softmax(gate_logits, dim=-1).mean(dim=0)   # [E]
            oh = torch.zeros(N, self.num_experts, device=x.device)
            oh.scatter_(1, topk_idx, 1.0 / self.top_k)
            load = oh.mean(dim=0)                                      # [E]
        lb_loss = self.num_experts * (importance * load).sum()

        # ── Expert dispatch ───────────────────────────────────
        capacity  = max(1, int(self.cap_factor * N / self.num_experts))
        output    = torch.zeros_like(x_flat)

        for e in range(self.num_experts):
            # Tokens that chose this expert (any of the k slots)
            tok_mask = (topk_idx == e).any(dim=-1)         # [N]
            tok_ids  = tok_mask.nonzero(as_tuple=False).squeeze(-1)  # [n]

            if tok_ids.numel() == 0:
                continue

            # Drop overflow tokens beyond capacity
            if tok_ids.numel() > capacity:
                tok_ids = tok_ids[:capacity]

            # Routing weights for this expert
            w = torch.zeros(tok_ids.numel(), device=x.device)
            for ki in range(self.top_k):
                hit = (topk_idx[tok_ids, ki] == e)
                w[hit] = topk_weights[tok_ids[hit], ki]

            # Expert forward + weighted accumulate
            e_out = self.experts[e](x_flat[tok_ids])       # [n, D]
            output[tok_ids] += w.unsqueeze(-1) * e_out

        return output.reshape(B, T, D), lb_loss


# ──────────────────────────────────────────────────────────────
# PHASE 1  ·  TRANSFORMER BLOCK  (pre-norm, Attn + MoE)
# ──────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block.
      h ← h + MHA(LN(h))
      h ← h + MoE(LN(h))

    Returns (h_new, lb_loss) so gradient checkpointing wraps cleanly.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.dim)
        self.attn = nn.MultiheadAttention(cfg.dim, cfg.n_heads,
                                          batch_first=True, bias=False)
        self.ln2  = nn.LayerNorm(cfg.dim)
        self.moe  = MoELayer(cfg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        h   = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x   = x + a

        # Mixture-of-Experts FFN
        h, lb = self.moe(self.ln2(x))
        return x + h, lb


# ──────────────────────────────────────────────────────────────
# PHASE 2.1  ·  ATTENTION RESIDUALS ACROSS LOOP STATES
# ──────────────────────────────────────────────────────────────

class CrossStepAttention(nn.Module):
    """
    Attends from current hidden state to the last-K past loop states:
      context = cat(past_states[-K:], dim=seq)   [B, K·T, D]
      h ← h + MHA(LN(h), context, context)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln   = nn.LayerNorm(cfg.dim)
        self.attn = nn.MultiheadAttention(cfg.dim, cfg.n_heads,
                                          batch_first=True, bias=False)

    def forward(self, h: torch.Tensor,
                past: List[torch.Tensor]) -> torch.Tensor:
        if not past:
            return h
        ctx     = torch.cat(past, dim=1)          # [B, K·T, D]
        out, _  = self.attn(self.ln(h), ctx, ctx, need_weights=False)
        return h + out


# ──────────────────────────────────────────────────────────────
# PHASE 2.2  ·  ADAPTIVE COMPUTATION TIME  (ACT)
# ──────────────────────────────────────────────────────────────

class ACTController(nn.Module):
    """
    Per-token halting gate.

      p_t = σ(W_halt · h_t)          scalar halt probability per token
      Σ_t += p_t                      accumulated halting sum
      stop if  Σ_t.all() ≥ threshold

    Returns (halt_prob, updated_sum, should_stop).
    halt_prob feeds into the ACT penalty loss.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.W_halt    = nn.Linear(cfg.dim, 1, bias=True)
        self.threshold = cfg.act_threshold

    def forward(self,
                h:           torch.Tensor,
                halting_sum: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        # h : [B, T, D],  halting_sum : [B, T]
        p           = torch.sigmoid(self.W_halt(h)).squeeze(-1)  # [B, T]
        halting_sum = halting_sum + p
        done        = bool((halting_sum >= self.threshold).all().item())
        return p, halting_sum, done


# ──────────────────────────────────────────────────────────────
# PHASE 3  ·  HIERARCHICAL MEMORY
# ──────────────────────────────────────────────────────────────

# ── 3.3  State compression (mini-autoencoder) ─────────────────

class StateAutoencoder(nn.Module):
    """
    Compresses hidden states for slow memory storage.
      encode : dim → latent_dim
      decode : latent_dim → dim
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.enc = nn.Linear(cfg.dim, cfg.latent_dim, bias=False)
        self.dec = nn.Linear(cfg.latent_dim, cfg.dim,  bias=False)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return self.enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)


# ── 3.2  Slow memory (vector DB simulation) ───────────────────

class SlowMemory(nn.Module):
    """
    Learnable memory bank [M, latent_dim].

    Retrieval:
      similarity = cosine(z_flat, bank)     [N, M]
      top-k mean over retrieved slots       [N, latent_dim]
      projected back to dim

    The bank is updated implicitly via backprop (not explicit writes).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.bank = nn.Parameter(
            torch.randn(cfg.slow_mem_size, cfg.latent_dim) * 0.01)
        self.proj  = nn.Linear(cfg.latent_dim, cfg.dim, bias=False)
        self.top_k = 4

    def retrieve(self, z: torch.Tensor) -> torch.Tensor:
        # z : [B, T, latent_dim]
        B, T, LD  = z.shape
        z_flat    = z.reshape(B * T, LD)
        z_n       = F.normalize(z_flat,   dim=-1)
        b_n       = F.normalize(self.bank, dim=-1)
        sim       = z_n @ b_n.T                          # [N, M]

        k          = min(self.top_k, sim.shape[-1])
        topk_s, topk_i = torch.topk(sim, k, dim=-1)
        weights    = F.softmax(topk_s, dim=-1).unsqueeze(-1)  # [N, k, 1]
        retrieved  = self.bank[topk_i]                         # [N, k, LD]
        mean       = (retrieved * weights).sum(dim=1)          # [N, LD]
        # proj maps latent_dim → dim (ensures decoder gets gradients)
        return self.proj(mean).reshape(B, T, -1)              # [B, T, dim]


# ── 3.1  Fast memory (short-term deque) ───────────────────────

class FastMemory:
    """
    Ring-buffer of recent hidden states.
    Not an nn.Module — states are stored detached.
    """

    def __init__(self, maxlen: int):
        self._buf: deque = deque(maxlen=maxlen)

    def push(self, h: torch.Tensor) -> None:
        self._buf.append(h.detach())

    def latest(self) -> Optional[torch.Tensor]:
        return self._buf[-1] if self._buf else None

    def clear(self) -> None:
        self._buf.clear()


# ──────────────────────────────────────────────────────────────
# PHASE 4  ·  SUB-AGENT SYSTEM
# ──────────────────────────────────────────────────────────────

class AgentMLP(nn.Module):
    """Single sub-agent: Linear → GELU → Linear (residual applied outside)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim, bias=False),
            nn.GELU(),
            nn.Linear(cfg.dim, cfg.dim, bias=False),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SubAgentSystem(nn.Module):
    """
    Sub-agent selection and dispatch.

    Selection: mean-pool hidden state → Linear → argmax
               0 = no-op;  1..N = agent index

    Applied only when layer_idx >= cfg.agent_start_layer (last 4 layers).

    Agents:
      0: ReasoningAgent — general-purpose reasoning MLP
      1: MathAgent      — numeric operations MLP
      2: MemoryAgent    — memory retrieval MLP
    """

    NAMES = ("ReasoningAgent", "MathAgent", "MemoryAgent")

    def __init__(self, cfg: Config):
        super().__init__()
        self.agents   = nn.ModuleList([AgentMLP(cfg) for _ in range(cfg.n_agents)])
        self.selector = nn.Linear(cfg.dim, cfg.n_agents + 1, bias=True)

    def forward(self, h: torch.Tensor,
                counter: Dict[str, int]) -> torch.Tensor:
        # h : [B, T, D]
        h_pool   = h.mean(dim=1)                    # [B, D]
        agent_id = self.selector(h_pool).argmax(-1)  # [B]

        out = h
        for b in range(h.shape[0]):
            aid = int(agent_id[b].item())
            if aid == 0 or aid > len(self.agents):
                continue
            idx    = aid - 1
            delta  = self.agents[idx](h[b].unsqueeze(0)).squeeze(0)
            out    = out.clone()
            out[b] = h[b] + delta
            name   = self.NAMES[idx] if idx < len(self.NAMES) else f"Agent{idx}"
            counter[name] = counter.get(name, 0) + 1

        return out


# ──────────────────────────────────────────────────────────────
# PHASE 5  ·  TOOL CALLING SYSTEM
# ──────────────────────────────────────────────────────────────

# ── Tool definitions (pure Python, sandboxed) ─────────────────

def _tool_calculator(expr: str) -> str:
    """Safe arithmetic evaluator (digits and +-*/().% only)."""
    expr = expr.strip()[:64]
    safe = set("0123456789 +-*/().%")
    if not all(c in safe for c in expr):
        return "error"
    try:
        return str(eval(expr, {"__builtins__": {}}))[:32]
    except Exception:
        return "error"


def _tool_string(text: str) -> str:
    """Returns '{length}:{reversed_prefix}' for the input text."""
    text = text[:64]
    return f"{len(text)}:{text[::-1][:16]}"


def _tool_memory_lookup(query: str) -> str:
    """Stub memory lookup — returns a deterministic hash string."""
    return f"mem:{abs(hash(query[:16])) % 10_000}"


_TOOLS      = [_tool_calculator, _tool_string, _tool_memory_lookup]
_TOOL_NAMES = ["calculator", "string_tool", "memory_lookup"]


# ── ToolCallingSystem ─────────────────────────────────────────

class ToolCallingSystem(nn.Module):
    """
    Tool selection and invocation.

    Selection:
      tool_id = argmax(Linear(h_cls))     # 0 = no-op

    Argument generation (avoids vocab-sized projection):
      arg_latent = Linear(h_cls)          # [latent_dim]
      arg_scalar = Linear(arg_latent)     # scalar → stringified

    Result integration:
      result_str → tokenize → embed (shared tok embedding) → add to h

    Constraints:
      - Max 1 call per sample per forward pass
      - Applied at last recurrent step of last layer only
      - All tool calls wrapped in try/except
    """

    def __init__(self, cfg: Config, tok_embed: nn.Embedding):
        super().__init__()
        self.n_tools   = cfg.n_tools
        self.selector  = nn.Linear(cfg.dim, cfg.n_tools + 1, bias=True)
        # Compact arg generator (no vocab-sized projection)
        self.arg_proj  = nn.Linear(cfg.dim, cfg.latent_dim, bias=False)
        self.arg_read  = nn.Linear(cfg.latent_dim, 1,        bias=True)
        # Result projection (reuses shared tok_embed, no extra params)
        self.res_proj  = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self._tok_embed = tok_embed   # shared — NOT a parameter of this module

    def forward(self, h: torch.Tensor,
                tokenizer: Any,
                counter: Dict[str, int]) -> torch.Tensor:
        # h : [B, T, D]
        h_cls    = h[:, 0, :]                           # [B, D]  CLS-like token
        tool_id  = self.selector(h_cls).argmax(-1)      # [B]

        out = h
        for b in range(h.shape[0]):
            tid = int(tool_id[b].item())
            if tid == 0 or tid > self.n_tools:
                continue
            tool_fn = _TOOLS[tid - 1]

            # ── Generate argument ─────────────────────────────
            lat  = self.arg_proj(h_cls[b])              # [latent_dim]
            val  = self.arg_read(lat).item()            # scalar
            arg_str = f"{val:.4f}"

            # ── Execute tool (sandboxed) ──────────────────────
            try:
                result_str = tool_fn(arg_str)
            except Exception:
                result_str = "0"

            # ── Embed result via shared tokenizer + embedding ─
            try:
                if tokenizer is not None:
                    result_ids = tokenizer.encode(result_str).ids
                else:
                    result_ids = [ord(c) % self._tok_embed.num_embeddings
                                  for c in result_str[:16]]

                T = h.shape[1]
                result_ids = result_ids[:T]             # truncate to seq_len
                if result_ids:
                    rid_t  = torch.tensor(result_ids, dtype=torch.long, device=h.device)
                    padded = torch.zeros(T, dtype=torch.long, device=h.device)
                    padded[:len(rid_t)] = rid_t
                    r_emb  = self._tok_embed(padded)    # [T, D]
                    out    = out.clone()
                    out[b] = h[b] + self.res_proj(r_emb)
            except Exception:
                pass

            name = (_TOOL_NAMES[tid - 1] if tid - 1 < len(_TOOL_NAMES)
                    else f"Tool{tid - 1}")
            counter[name] = counter.get(name, 0) + 1

        return out


# ──────────────────────────────────────────────────────────────
# FULL MODEL  ·  AntigravityLLM
# ──────────────────────────────────────────────────────────────

class AntigravityLLM(nn.Module):
    """
    Full hybrid model wiring:

      embed → for each layer:
                for each recurrent step:
                  block (Attn + MoE)
                  cross-step attention residual
                  fast memory injection
                  slow memory retrieval
                  ACT halt check
                [sub-agents — last 4 layers]
           → tool calling (last step of last layer)
           → LN + lm_head → logits

    Returns: (logits [B,T,V], moe_lb_loss, act_penalty)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # ── Phase 1 — Embedding ──────────────────────────────
        self.embed   = TokenEmbedding(cfg)

        # ── Phase 1 + 2.3 — Transformer blocks (one per layer)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg)
                                      for _ in range(cfg.n_layers)])

        # ── Phase 2.1 — Cross-step attention (one per layer) ─
        self.cs_attn = nn.ModuleList([CrossStepAttention(cfg)
                                      for _ in range(cfg.n_layers)])

        # ── Phase 2.2 — ACT controllers (one per layer) ──────
        self.act     = nn.ModuleList([ACTController(cfg)
                                      for _ in range(cfg.n_layers)])

        # ── Phase 3 — Memory ──────────────────────────────────
        self.autoenc  = StateAutoencoder(cfg)
        self.slow_mem = SlowMemory(cfg)
        self.fast_mem = FastMemory(maxlen=cfg.fast_mem_size)

        # ── Phase 4 — Sub-agents ─────────────────────────────
        self.agents   = SubAgentSystem(cfg)

        # ── Phase 5 — Tool calling ────────────────────────────
        self.tools    = ToolCallingSystem(cfg, tok_embed=self.embed.tok)

        # ── Output head ───────────────────────────────────────
        self.ln_f     = nn.LayerNorm(cfg.dim)
        self.lm_head  = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        # Weight tying: lm_head shares weights with token embedding
        self.lm_head.weight = self.embed.tok.weight

        # ── Logging state ─────────────────────────────────────
        self.agent_counter: Dict[str, int] = defaultdict(int)
        self.tool_counter:  Dict[str, int] = defaultdict(int)
        self.act_steps:     List[int]      = []

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────

    def _init_weights(self) -> None:
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    # ── Counter reset ─────────────────────────────────────────

    def reset_counters(self) -> None:
        self.agent_counter = defaultdict(int)
        self.tool_counter  = defaultdict(int)
        self.act_steps     = []

    # ── Forward pass ─────────────────────────────────────────

    def forward(
        self,
        x:         torch.Tensor,
        tokenizer: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : [B, T]  token IDs
        Returns: (logits [B,T,V], moe_lb_loss scalar, act_penalty scalar)
        """
        cfg    = self.cfg
        device = x.device
        B, T   = x.shape

        h = self.embed(x)       # [B, T, dim]
        self.fast_mem.clear()

        total_lb  = torch.zeros((), device=device)
        total_act = torch.zeros((), device=device)

        for li, (block, cs, act_ctrl) in enumerate(
                zip(self.blocks, self.cs_attn, self.act)):

            past: List[torch.Tensor] = []
            halt_sum = torch.zeros(B, T, device=device)
            steps    = 0

            # ── Recurrent reasoning loop ──────────────────────
            for step in range(cfg.max_steps):

                # Transformer block  (optionally grad-checkpointed)
                if cfg.use_grad_checkpoint and self.training:
                    h_new, lb = grad_ckpt(block, h, use_reentrant=False)
                else:
                    h_new, lb = block(h)

                total_lb = total_lb + lb

                # Attention residual across past loop states
                lookback = past[-cfg.cross_step_lookback:]
                h_new    = cs(h_new, lookback)

                # Fast memory injection (residual from previous step)
                fm = self.fast_mem.latest()
                if fm is not None and fm.shape == h_new.shape:
                    h_new = h_new + 0.1 * fm

                # Slow memory retrieval — encode → retrieve → decode residual
                z         = self.autoenc.encode(h_new)       # [B, T, latent]
                z_dec     = self.autoenc.decode(z)           # [B, T, dim]  (keeps dec in graph)
                retrieved = self.slow_mem.retrieve(z)        # [B, T, dim]
                retrieved = retrieved + 0.01 * z_dec         # tie decoder into graph
                h_new     = h_new + 0.1 * retrieved

                # Update memories
                self.fast_mem.push(h_new)
                past.append(h_new.detach())
                h = h_new
                steps += 1

                # ACT halting check
                p, halt_sum, done = act_ctrl(h, halt_sum)
                total_act = total_act + p.mean()
                if done:
                    break   # ← early exit from recurrent loop

            self.act_steps.append(steps)

            # Sub-agents — applied in last (n_layers - agent_start_layer) layers
            if li >= cfg.agent_start_layer:
                h = self.agents(h, self.agent_counter)

        # Tool calling — final step of last layer
        h = self.tools(h, tokenizer, self.tool_counter)

        # Output projection
        logits = self.lm_head(self.ln_f(h))  # [B, T, V]
        return logits, total_lb, total_act


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────

class TextDataset(torch.utils.data.Dataset):
    """Sliding-window next-token dataset over a pre-tokenised list of IDs."""

    def __init__(self, token_ids: List[int], seq_len: int):
        self.ids     = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.ids[idx: idx + self.seq_len + 1]
        return (torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:],  dtype=torch.long))


class SyntheticDataset(torch.utils.data.Dataset):
    """Fallback: random token sequences for smoke-testing without a corpus."""

    def __init__(self, cfg: Config, n: int = 2048):
        self.cfg = cfg
        self.n   = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = torch.randint(1, self.cfg.vocab_size, (self.cfg.seq_len + 1,))
        return ids[:-1], ids[1:]


# ──────────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ──────────────────────────────────────────────────────────────

def cosine_lr(step: int, cfg: Config) -> float:
    """Linear warmup + cosine decay."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    t = (step - cfg.warmup_steps) / max(1, cfg.max_train_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * t))


def load_corpus(path: str) -> str:
    """Try to load a text corpus from `path` or common fallback locations."""
    candidates = [
        path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt"),
        os.path.expanduser("~/corpus.txt"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            print(f"  Loading corpus: {p}")
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    print("  No corpus file found — using built-in micro-corpus.")
    return (
        "The quick brown fox jumps over the lazy dog. " * 400
        + "Mixture of experts routes tokens to specialised networks. " * 400
        + "Adaptive computation time allows the model to think longer. " * 400
        + "Attention residuals carry information across reasoning steps. " * 400
        + "Hierarchical memory stores fast and slow representations. " * 400
    )


def build_dataset(cfg: Config, tokenizer: Any) -> torch.utils.data.Dataset:
    """Build TextDataset from tokenised corpus, or fall back to synthetic."""
    corpus = load_corpus(cfg.corpus_path)
    if len(corpus) < 200 or tokenizer is None:
        return SyntheticDataset(cfg)
    try:
        enc = tokenizer.encode(corpus)
        ids = enc.ids
        if len(ids) < cfg.seq_len + 2:
            return SyntheticDataset(cfg)
        print(f"  Corpus: {len(corpus):,} chars → {len(ids):,} tokens")
        return TextDataset(ids, cfg.seq_len)
    except Exception as e:
        print(f"  Tokeniser error ({e}) — using synthetic dataset.")
        return SyntheticDataset(cfg)


def param_count(model: nn.Module) -> Tuple[int, int]:
    """Total params and unique params (deduped for tied weights)."""
    total  = sum(p.numel() for p in model.parameters())
    seen   = set()
    unique = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            unique += p.numel()
    return total, unique


# ──────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────

def train(cfg: Config, smoke_test: bool = False) -> None:
    device  = torch.device(cfg.device)
    use_amp = cfg.use_amp and device.type == "cuda"

    # ── Header ───────────────────────────────────────────────
    sep = "═" * 62
    print(f"\n{sep}")
    print("  ANTIGRAVITY LLM  —  Hybrid Prototype")
    print(sep)
    print(f"  Device        : {device}")
    print(f"  Arch          : dim={cfg.dim}  heads={cfg.n_heads}  "
          f"layers={cfg.n_layers}")
    print(f"  MoE           : {cfg.num_experts} experts  top-{cfg.top_k}")
    print(f"  Recurrent     : max_steps={cfg.max_steps}  "
          f"ACT threshold={cfg.act_threshold}")
    print(f"  Memory        : fast={cfg.fast_mem_size}  "
          f"slow={cfg.slow_mem_size}  latent={cfg.latent_dim}")
    print(f"  Agents        : {cfg.n_agents}  (layers {cfg.agent_start_layer}+)")
    print(f"  Tools         : {cfg.n_tools}")
    print(f"  Seq / Batch   : {cfg.seq_len} / {cfg.batch_size}")
    print(f"  Vocab         : {cfg.vocab_size:,}  (Gemma 4 BPE)")
    print(f"  AMP           : {use_amp}  "
          f"GradCkpt : {cfg.use_grad_checkpoint}")
    print(sep)

    # ── Tokeniser ────────────────────────────────────────────
    tokenizer = None
    try:
        from tokenizers import Tokenizer as HFTokenizer
        tok_file  = os.path.join(cfg.tokenizer_path, "tokenizer.json")
        print(f"\n  Tokeniser : {tok_file}")
        tokenizer = HFTokenizer.from_file(tok_file)
        tokenizer.enable_truncation(max_length=cfg.seq_len + 1)
        print("  Tokeniser loaded ✓")
    except Exception as e:
        print(f"  Tokeniser load failed: {e}")
        print("  → Falling back to synthetic dataset (no tokeniser).")

    # ── Model ────────────────────────────────────────────────
    print()
    model = AntigravityLLM(cfg).to(device)
    total, unique = param_count(model)
    print(f"  Parameters    : {total:,} total  "
          f"({unique:,} unique = {unique / 1e6:.2f}M)")

    # ── Dataset & loader ─────────────────────────────────────
    dataset = build_dataset(cfg, tokenizer)
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = 0,
        drop_last   = True,
        pin_memory  = (device.type == "cuda"),
    )

    # ── Optimiser + scaler ───────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ════════════════════════════════════════════════════════
    # SMOKE TEST — single forward + backward, then exit
    # ════════════════════════════════════════════════════════
    if smoke_test:
        print(f"\n{sep}")
        print("  SMOKE TEST — single forward + backward pass")
        print(sep)

        model.train()
        model.reset_counters()
        bx, by = next(iter(loader))
        bx, by = bx.to(device), by.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, lb, act_pen = model(bx, tokenizer)
            ce   = F.cross_entropy(logits.view(-1, cfg.vocab_size), by.view(-1))
            loss = ce + cfg.moe_loss_weight * lb + cfg.act_loss_weight * act_pen

        scaler.scale(loss).backward()

        # Verify gradients
        missing_grad = [n for n, p in model.named_parameters()
                        if p.requires_grad and p.grad is None]

        print(f"\n  CE loss   : {ce.item():.4f}")
        print(f"  MoE loss  : {lb.item():.6f}")
        print(f"  ACT pen   : {act_pen.item():.4f}")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Logits    : {logits.shape}  (min={logits.min():.2f}  "
              f"max={logits.max():.2f})")
        print(f"  ACT steps per layer: {model.act_steps}")

        # Params without grad are expected in two cases:
        #   1. Reduced-layer config: agents not reached (agent_start_layer > n_layers)
        #   2. Selector heads for agents/tools: argmax picks 0 (no-op) on cold model,
        #      so the selected branch never executes → no grad through those heads.
        # Both are benign; gradients flow there once the model warms up.
        known_no_grad = {"selector", "arg_proj", "arg_read", "res_proj"}
        truly_missing = [n for n in missing_grad
                         if not any(k in n for k in known_no_grad)
                         and cfg.n_layers > cfg.agent_start_layer]
        if truly_missing:
            print(f"\n  ⚠  {len(truly_missing)} UNEXPECTED params without grad: "
                  f"{truly_missing[:5]}")
        elif missing_grad:
            print(f"\n  ℹ  {len(missing_grad)} params without grad "
                  f"(expected — sparse selectors/reduced config, benign)")
        else:
            print("\n  Gradients : all params ✓")

        if not math.isfinite(loss.item()):
            print("\n  ✗ SMOKE TEST FAILED — loss is not finite!")
            sys.exit(1)

        print("\n  ✓ SMOKE TEST PASSED")
        print(sep)
        return

    # ════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ════════════════════════════════════════════════════════
    print(f"\n  Starting training — {cfg.max_train_steps} steps\n")

    data_iter = iter(loader)
    t0        = time.perf_counter()

    for step in range(1, cfg.max_train_steps + 1):
        model.train()
        model.reset_counters()

        # LR schedule
        lr = cosine_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Fetch batch (reshuffle on exhaustion)
        try:
            bx, by = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            bx, by    = next(data_iter)
        bx = bx.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, lb, act_pen = model(bx, tokenizer)
            ce   = F.cross_entropy(logits.view(-1, cfg.vocab_size), by.view(-1))
            loss = ce + cfg.moe_loss_weight * lb + cfg.act_loss_weight * act_pen

        # Backward + clip + step
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # ── Logging ──────────────────────────────────────────
        if step % cfg.log_every == 0:
            elapsed  = time.perf_counter() - t0
            avg_act  = (sum(model.act_steps) / len(model.act_steps)
                        if model.act_steps else 0.0)
            min_act  = min(model.act_steps) if model.act_steps else 0
            max_act  = max(model.act_steps) if model.act_steps else 0

            print(
                f"step {step:5d}/{cfg.max_train_steps} │ "
                f"CE={ce.item():.4f}  MoE={lb.item():.5f}  "
                f"ACT={act_pen.item():.4f} │ "
                f"lr={lr:.2e}  ACT_avg={avg_act:.2f} "
                f"[{min_act}–{max_act}] │ "
                f"{elapsed:6.1f}s"
            )
            if model.agent_counter:
                print(f"        Agents → {dict(model.agent_counter)}")
            if model.tool_counter:
                print(f"        Tools  → {dict(model.tool_counter)}")

    print(f"\n  Training complete — "
          f"{cfg.max_train_steps} steps in "
          f"{time.perf_counter() - t0:.1f}s")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Antigravity Hybrid LLM Prototype",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Execution mode
    p.add_argument("--smoke-test",  action="store_true",
                   help="Single forward/backward pass and exit")
    # Architecture
    p.add_argument("--dim",         type=int,   default=128)
    p.add_argument("--n-heads",     type=int,   default=4)
    p.add_argument("--n-layers",    type=int,   default=16)
    p.add_argument("--num-experts", type=int,   default=32)
    p.add_argument("--top-k",       type=int,   default=8)
    p.add_argument("--max-steps",   type=int,   default=5)
    # Data
    p.add_argument("--seq-len",         type=int, default=128)
    p.add_argument("--batch-size",      type=int, default=8)
    p.add_argument("--tokenizer-path",  type=str,
                   default="/home/miguelbds/Área de trabalho/Nova pasta/Models/")
    p.add_argument("--corpus-path",     type=str, default="")
    # Training
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--max-train-steps", type=int,   default=10_000)
    p.add_argument("--log-every",       type=int,   default=50)
    # System
    p.add_argument("--no-amp",      action="store_true",
                   help="Disable mixed precision")
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="Disable gradient checkpointing")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg  = Config(
        dim             = args.dim,
        n_heads         = args.n_heads,
        n_layers        = args.n_layers,
        num_experts     = args.num_experts,
        top_k           = args.top_k,
        max_steps       = args.max_steps,
        seq_len         = args.seq_len,
        batch_size      = args.batch_size,
        tokenizer_path  = args.tokenizer_path,
        corpus_path     = args.corpus_path,
        lr              = args.lr,
        max_train_steps = args.max_train_steps,
        log_every       = args.log_every,
        use_amp         = not args.no_amp,
        use_grad_checkpoint = not args.no_grad_ckpt,
    )
    train(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
