"""
LLM-MoE Configuration
=====================
Dataclass-based configuration for the Transformer Decoder-only model
with Mixture of Experts.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Core model architecture configuration."""

    # --- Vocabulary & Embedding ---
    vocab_size: int = 4096
    max_seq_len: int = 512

    # --- Transformer dimensions ---
    dim: int = 256                # Hidden dimension (d_model)
    n_layers: int = 4             # Number of transformer layers
    n_heads: int = 4              # Number of attention heads
    head_dim: int = 64            # Dimension per head (dim // n_heads)

    # --- Mixture of Experts ---
    n_experts: int = 4            # Total number of experts
    n_active_experts: int = 2     # Top-K experts selected per token
    expert_dim: int = 512         # FFN intermediate dimension per expert (typically 2-4x dim)

    # --- Normalization ---
    norm_eps: float = 1e-6        # RMSNorm epsilon

    # --- RoPE ---
    rope_theta: float = 10000.0   # RoPE base frequency

    # --- Special tokens ---
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    # --- Reasoning tokens (IDs assigned during tokenizer training) ---
    think_token: str = "<think>"
    step_token: str = "<step>"
    verify_token: str = "<verify>"
    conclude_token: str = "<conclude>"
    end_think_token: str = "</think>"

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        self.head_dim = self.dim // self.n_heads


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # --- Optimization ---
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- Schedule ---
    warmup_steps: int = 100
    max_steps: int = 5000
    eval_interval: int = 100
    log_interval: int = 10

    # --- Batch ---
    batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # --- Mixed Precision ---
    use_amp: bool = True
    amp_dtype: str = "float16"    # "float16" or "bfloat16"

    # --- Gradient Checkpointing ---
    gradient_checkpointing: bool = True

    # --- Load balancing ---
    moe_aux_loss_weight: float = 0.01  # Weight for MoE load balancing loss

    # --- Paths ---
    data_path: str = "../data"              # Directory with text files (all formats)
    checkpoint_dir: str = "../checkpoints"
    export_path: str = "../model.bin"
    tokenizer_path: str = "../tokenizer"

    # --- Web Scraping ---
    enable_scraping: bool = False           # Enable web scraping for training data
    scrape_urls: List[str] = field(default_factory=list)  # Seed URLs to scrape
    scrape_url_file: Optional[str] = None   # File with URLs (one per line)
    scrape_max_pages: int = 50              # Max pages to scrape
    scrape_max_depth: int = 2               # Max crawl depth
    scrape_delay: float = 1.0               # Delay between requests (seconds)
    scrape_follow_links: bool = True        # Follow links from seed pages
    scrape_stay_on_domain: bool = True      # Only follow same-domain links


@dataclass
class ExportConfig:
    """Export and quantization configuration."""

    # --- Quantization ---
    quantize_weights: bool = True
    quantize_bits: int = 4         # INT4
    quantize_embeddings: bool = False  # Keep embeddings in FP32

    # --- Scales ---
    scale_mode: str = "per_tensor"  # "per_tensor" or "per_channel"


# ── Presets ──────────────────────────────────────────────────────────────

DEMO_CONFIG = ModelConfig(
    vocab_size=4096,
    max_seq_len=512,
    dim=256,
    n_layers=4,
    n_heads=4,
    n_experts=4,
    n_active_experts=2,
    expert_dim=512,
)

FULL_CONFIG = ModelConfig(
    vocab_size=32000,
    max_seq_len=2048,
    dim=512,
    n_layers=128,
    n_heads=8,
    n_experts=8,
    n_active_experts=2,
    expert_dim=1408,
)
