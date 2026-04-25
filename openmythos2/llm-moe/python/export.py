"""
Model Export — INT4 Quantization + Binary Format
=================================================
Exports trained PyTorch model to a custom binary format
optimized for the C runtime.

Features:
- INT4 quantization with per-tensor scale
- 2 values packed per byte
- FP32 for embeddings (preserves precision)
- Complete model + tokenizer in single binary
"""

import os
import sys
import struct
import argparse
from typing import Dict, Tuple

import numpy as np
import torch

from config import ModelConfig, ExportConfig, DEMO_CONFIG, FULL_CONFIG
from tokenizer import BPETokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Binary Format Constants
# ═══════════════════════════════════════════════════════════════════════════

MAGIC = b"LLMO"
FORMAT_VERSION = 1

# Quantization types
QUANT_NONE = 0      # FP32
QUANT_INT4 = 1      # INT4 packed (2 per byte)


# ═══════════════════════════════════════════════════════════════════════════
# INT4 Quantization
# ═══════════════════════════════════════════════════════════════════════════

def quantize_int4(tensor: torch.Tensor) -> Tuple[bytes, float]:
    """
    Quantize a tensor to INT4 with per-tensor scale.

    INT4 range: [-8, 7] (4 bits signed)
    Scale: max(abs(tensor)) / 7

    Packing: 2 values per byte
        byte = (val1 & 0x0F) | (val2 << 4)
        where val1 and val2 are biased to [0, 15]: stored = value + 8

    Args:
        tensor: FP32 tensor to quantize

    Returns:
        packed_data: bytes with 2 INT4 values per byte
        scale: float32 scale factor
    """
    flat = tensor.flatten().float()
    n = flat.numel()

    # Compute scale
    amax = flat.abs().max().item()
    scale = amax / 7.0 if amax > 0 else 1.0

    # Quantize to [-8, 7]
    quantized = torch.clamp(torch.round(flat / scale), -8, 7).to(torch.int8)

    # Bias to [0, 15] for unsigned packing
    biased = (quantized + 8).to(torch.uint8)

    # Pack 2 values per byte
    # Pad to even length
    if n % 2 != 0:
        biased = torch.cat([biased, torch.zeros(1, dtype=torch.uint8)])

    low = biased[0::2]   # even indices
    high = biased[1::2]  # odd indices
    packed = (low & 0x0F) | (high << 4)

    return packed.numpy().tobytes(), scale


def write_tensor_fp32(f, name: str, tensor: torch.Tensor):
    """Write a tensor in FP32 format."""
    data = tensor.detach().cpu().float().numpy()
    name_bytes = name.encode("utf-8")
    shape = list(data.shape)

    # Name
    f.write(struct.pack("<H", len(name_bytes)))
    f.write(name_bytes)

    # Type
    f.write(struct.pack("<B", QUANT_NONE))

    # Shape
    f.write(struct.pack("<B", len(shape)))
    for s in shape:
        f.write(struct.pack("<I", s))

    # Data
    f.write(data.tobytes())


def write_tensor_int4(f, name: str, tensor: torch.Tensor):
    """Write a tensor in INT4 packed format."""
    name_bytes = name.encode("utf-8")
    shape = list(tensor.shape)
    numel = tensor.numel()

    packed_data, scale = quantize_int4(tensor)

    # Name
    f.write(struct.pack("<H", len(name_bytes)))
    f.write(name_bytes)

    # Type
    f.write(struct.pack("<B", QUANT_INT4))

    # Shape
    f.write(struct.pack("<B", len(shape)))
    for s in shape:
        f.write(struct.pack("<I", s))

    # Scale
    f.write(struct.pack("<f", scale))

    # Number of elements (needed for unpacking)
    f.write(struct.pack("<I", numel))

    # Packed data
    f.write(packed_data)


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════

def export_model(
    checkpoint_path: str,
    output_path: str,
    tokenizer_path: str,
    export_cfg: ExportConfig = ExportConfig(),
):
    """
    Export trained model to binary format.

    Binary layout:
        [MAGIC: 4B] [VERSION: 4B]
        [CONFIG SECTION]
        [TOKENIZER SECTION]
        [WEIGHTS SECTION]
    """
    print(f"[Export] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_cfg: ModelConfig = checkpoint["model_config"]
    state_dict: Dict[str, torch.Tensor] = checkpoint["model_state_dict"]

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    print(f"[Export] Model config: dim={model_cfg.dim}, layers={model_cfg.n_layers}, "
          f"experts={model_cfg.n_experts}, vocab={model_cfg.vocab_size}")
    print(f"[Export] Quantization: {'INT4' if export_cfg.quantize_weights else 'FP32'}")

    with open(output_path, "wb") as f:
        # ── Header ──
        f.write(MAGIC)
        f.write(struct.pack("<I", FORMAT_VERSION))

        # ── Config ──
        f.write(struct.pack("<I", model_cfg.vocab_size))
        f.write(struct.pack("<I", model_cfg.max_seq_len))
        f.write(struct.pack("<I", model_cfg.dim))
        f.write(struct.pack("<I", model_cfg.n_layers))
        f.write(struct.pack("<I", model_cfg.n_heads))
        f.write(struct.pack("<I", model_cfg.head_dim))
        f.write(struct.pack("<I", model_cfg.n_experts))
        f.write(struct.pack("<I", model_cfg.n_active_experts))
        f.write(struct.pack("<I", model_cfg.expert_dim))
        f.write(struct.pack("<f", model_cfg.norm_eps))
        f.write(struct.pack("<f", model_cfg.rope_theta))

        # Special token IDs
        f.write(struct.pack("<I", model_cfg.pad_id))
        f.write(struct.pack("<I", model_cfg.unk_id))
        f.write(struct.pack("<I", model_cfg.bos_id))
        f.write(struct.pack("<I", model_cfg.eos_id))

        # ── Tokenizer ──
        f.write(struct.pack("<I", tokenizer.vocab_size))
        f.write(struct.pack("<I", len(tokenizer.merges)))

        # Vocab entries (sorted by ID)
        for token_str, token_id in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
            token_bytes = token_str.encode("utf-8")
            f.write(struct.pack("<H", len(token_bytes)))
            f.write(token_bytes)
            f.write(struct.pack("<I", token_id))

        # Merge rules
        for a, b in tokenizer.merges:
            a_bytes = a.encode("utf-8")
            b_bytes = b.encode("utf-8")
            f.write(struct.pack("<H", len(a_bytes)))
            f.write(a_bytes)
            f.write(struct.pack("<H", len(b_bytes)))
            f.write(b_bytes)

        # ── Weights ──
        n_tensors = len(state_dict)
        f.write(struct.pack("<I", n_tensors))

        total_original = 0
        total_quantized = 0

        for name, tensor in state_dict.items():
            tensor = tensor.detach().cpu()
            original_size = tensor.numel() * 4  # FP32 = 4 bytes
            total_original += original_size

            # Decide quantization
            should_quantize = (
                export_cfg.quantize_weights
                and tensor.ndim >= 2
                and "embedding" not in name
                and "norm" not in name
            )

            if not export_cfg.quantize_embeddings and "embedding" in name:
                should_quantize = False

            if should_quantize:
                write_tensor_int4(f, name, tensor)
                # INT4: numel/2 bytes + 4 (scale)
                q_size = (tensor.numel() + 1) // 2 + 4
                total_quantized += q_size
            else:
                write_tensor_fp32(f, name, tensor)
                total_quantized += original_size

        file_size = f.tell()

    print(f"\n[Export] Complete!")
    print(f"  Original size:  {total_original / 1e6:.1f} MB")
    print(f"  Quantized size: {total_quantized / 1e6:.1f} MB")
    print(f"  File size:      {file_size / 1e6:.1f} MB")
    print(f"  Compression:    {total_original / total_quantized:.1f}x")
    print(f"  Output: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LLM-MoE to binary format")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default="../model.bin", help="Output binary path"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="../tokenizer", help="Tokenizer directory"
    )
    parser.add_argument(
        "--no-quantize", action="store_true", help="Disable INT4 quantization"
    )
    parser.add_argument(
        "--quantize-embeddings",
        action="store_true",
        help="Also quantize embedding weights",
    )
    args = parser.parse_args()

    export_cfg = ExportConfig(
        quantize_weights=not args.no_quantize,
        quantize_embeddings=args.quantize_embeddings,
    )

    export_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
        export_cfg=export_cfg,
    )
