"""
Training Pipeline
=================
Complete training loop for the Transformer-MoE language model.

Features:
- Mixed Precision (FP16/BF16)
- Gradient Checkpointing
- AdamW with Warmup + Cosine Decay
- MoE load balancing loss
- Checkpoint saving
- Simple text dataset
"""

import os
import sys
import math
import time
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import ModelConfig, TrainConfig, DEMO_CONFIG, FULL_CONFIG
from tokenizer import BPETokenizer
from model import TransformerLM


# Supported text file extensions
TEXT_EXTENSIONS = {
    # Natural language
    '.txt', '.md', '.rst', '.tex', '.log',
    # Programming languages
    '.py', '.c', '.h', '.cpp', '.hpp', '.cc',
    '.js', '.ts', '.jsx', '.tsx',
    '.java', '.kt', '.scala',
    '.rs', '.go', '.rb', '.lua', '.zig',
    '.cs', '.swift', '.r', '.jl',
    '.php', '.pl', '.ex', '.exs',
    # Config & data formats
    '.json', '.xml', '.yaml', '.yml', '.toml',
    '.ini', '.cfg', '.conf', '.env',
    '.csv', '.tsv',
    # Web
    '.html', '.htm', '.css', '.scss', '.sass', '.less',
    '.svg',
    # Shell & DevOps
    '.sh', '.bash', '.zsh', '.fish',
    '.dockerfile', '.tf', '.hcl',
    # Database
    '.sql',
    # Docs
    '.org', '.adoc',
}


def load_text_files(data_path: str) -> str:
    """
    Recursively load all text-based files from a directory (or a single file).
    Returns the concatenated text with file separators.
    """
    if os.path.isfile(data_path):
        with open(data_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    if not os.path.isdir(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")

    all_texts = []
    file_count = 0
    skipped = 0

    for root, dirs, files in os.walk(data_path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

        for fname in sorted(files):
            # Check extension
            _, ext = os.path.splitext(fname.lower())

            # Also support Makefile, Dockerfile, etc. (no extension)
            is_special = fname.lower() in {
                'makefile', 'dockerfile', 'jenkinsfile', 'vagrantfile',
                'gemfile', 'rakefile', 'procfile', 'cmakelists.txt',
            }

            if ext not in TEXT_EXTENSIONS and not is_special:
                skipped += 1
                continue

            filepath = os.path.join(root, fname)
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                if len(content.strip()) > 0:
                    # Add a separator between files for context boundaries
                    all_texts.append(f"\n\n--- {fname} ---\n\n{content}")
                    file_count += 1

            except (IOError, OSError) as e:
                print(f"  [Warning] Could not read {filepath}: {e}")

    print(f"[DataLoader] Loaded {file_count} files, skipped {skipped} non-text files")

    if not all_texts:
        raise ValueError(f"No text files found in {data_path}")

    return "\n".join(all_texts)


class TextDataset(Dataset):
    """
    Text dataset that loads ALL text-based files from a directory,
    tokenizes the entire corpus, and returns fixed-length chunks
    for language modeling.

    Supported formats: Python, C, JS/TS, SQL, JSON, XML, YAML, HTML,
    Markdown, Bash, Rust, Go, Java, Ruby, and many more.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: BPETokenizer,
        seq_len: int,
    ):
        super().__init__()
        self.seq_len = seq_len

        print(f"[Dataset] Loading from: {data_path}")
        text = load_text_files(data_path)
        print(f"[Dataset] Total text: {len(text):,} characters")

        # Tokenize entire corpus
        self.tokens = tokenizer.encode(text, add_special=False)
        print(f"[Dataset] {len(self.tokens):,} tokens, seq_len={seq_len}")

        # Calculate number of complete sequences
        self.n_samples = max(1, (len(self.tokens) - 1) // seq_len)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target shift

        chunk = self.tokens[start:end]

        # Pad if necessary
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ═══════════════════════════════════════════════════════════════════════════
# Learning Rate Schedule
# ═══════════════════════════════════════════════════════════════════════════

def get_lr(step: int, train_cfg: TrainConfig) -> float:
    """
    Learning rate schedule: linear warmup + cosine decay.
    """
    if step < train_cfg.warmup_steps:
        # Linear warmup
        return train_cfg.learning_rate * (step + 1) / train_cfg.warmup_steps

    # Cosine decay
    decay_steps = train_cfg.max_steps - train_cfg.warmup_steps
    current = step - train_cfg.warmup_steps
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * current / decay_steps))
    return train_cfg.min_learning_rate + (
        train_cfg.learning_rate - train_cfg.min_learning_rate
    ) * cosine_decay


# ═══════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
):
    """Main training function."""

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Web Scraping (optional) ──
    if train_cfg.enable_scraping:
        print("\n[Train] Web scraping enabled — fetching training data from the web...")
        try:
            from scraper import WebScraper, ScraperConfig

            scraper_cfg = ScraperConfig(
                seed_urls=train_cfg.scrape_urls,
                url_file=train_cfg.scrape_url_file,
                max_pages=train_cfg.scrape_max_pages,
                max_depth=train_cfg.scrape_max_depth,
                delay_seconds=train_cfg.scrape_delay,
                follow_links=train_cfg.scrape_follow_links,
                stay_on_domain=train_cfg.scrape_stay_on_domain,
                output_dir=os.path.join(train_cfg.data_path, "scraped"),
            )

            scraper = WebScraper(scraper_cfg)
            results = scraper.scrape()

            if results:
                scrape_dir = scraper.save(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        train_cfg.data_path, "scraped"
                    )
                )
                print(f"[Train] Scraped {len(results)} pages, "
                      f"saved to {scrape_dir}")
            else:
                print("[Train] No pages scraped, continuing with existing data")

        except ImportError:
            print("[Train] WARNING: Web scraping requires 'requests' and 'beautifulsoup4'")
            print("[Train]   Install with: pip install requests beautifulsoup4")
            print("[Train]   Continuing without web data...")
        except Exception as e:
            print(f"[Train] WARNING: Scraping failed: {e}")
            print("[Train]   Continuing without web data...")

    # ── Tokenizer ──
    tokenizer = BPETokenizer()
    tokenizer_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), train_cfg.tokenizer_path
    )

    if os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
        tokenizer.load(tokenizer_path)
    else:
        print("[Train] Training tokenizer on all text files...")
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), train_cfg.data_path
        )
        text = load_text_files(data_path)
        tokenizer.train(text, vocab_size=model_cfg.vocab_size)
        tokenizer.save(tokenizer_path)
        tokenizer.export_binary(os.path.join(tokenizer_path, "tokenizer.bin"))

    # Update vocab size to match tokenizer
    model_cfg.vocab_size = tokenizer.vocab_size

    # ── Dataset ──
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), train_cfg.data_path
    )
    dataset = TextDataset(data_path, tokenizer, model_cfg.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # ── Model ──
    model = TransformerLM(model_cfg).to(device)

    if train_cfg.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    total_params = model.count_parameters()
    active_params = model.count_active_parameters()
    print(f"[Train] Total parameters: {total_params:,}")
    print(f"[Train] Active parameters per token: {active_params:,}")
    print(f"[Train] Model config: {model_cfg}")

    # ── Optimizer ──
    # Separate weight decay groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        fused=device.type == "cuda",
    )

    # ── Mixed Precision ──
    amp_dtype = torch.float16 if train_cfg.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(train_cfg.use_amp and device.type == "cuda"))
    autocast_ctx = torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=train_cfg.use_amp,
    )

    # ── Training Loop ──
    print(f"\n[Train] Starting training for {train_cfg.max_steps} steps")
    print(f"  batch_size={train_cfg.batch_size}")
    print(f"  grad_accum={train_cfg.gradient_accumulation_steps}")
    print(f"  effective_batch={train_cfg.batch_size * train_cfg.gradient_accumulation_steps}")

    model.train()
    step = 0
    running_loss = 0.0
    data_iter = iter(dataloader)
    t0 = time.time()

    while step < train_cfg.max_steps:
        # Update learning rate
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_aux_loss = 0.0

        for micro_step in range(train_cfg.gradient_accumulation_steps):
            # Get batch (cycle through data)
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            # Forward
            with autocast_ctx:
                logits, _, aux_loss = model(x)
                # Language modeling loss
                lm_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=0,  # ignore padding
                )
                # Total loss
                loss = lm_loss + train_cfg.moe_aux_loss_weight * aux_loss
                loss = loss / train_cfg.gradient_accumulation_steps

            # Backward
            scaler.scale(loss).backward()

            accum_loss += lm_loss.item() / train_cfg.gradient_accumulation_steps
            accum_aux_loss += aux_loss.item() / train_cfg.gradient_accumulation_steps

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        running_loss += accum_loss
        step += 1

        # ── Logging ──
        if step % train_cfg.log_interval == 0:
            avg_loss = running_loss / train_cfg.log_interval
            elapsed = time.time() - t0
            tokens_per_sec = (
                train_cfg.batch_size
                * train_cfg.gradient_accumulation_steps
                * model_cfg.max_seq_len
                * train_cfg.log_interval
                / elapsed
            )
            print(
                f"  step {step:5d}/{train_cfg.max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"aux_loss {accum_aux_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {grad_norm:.2f} | "
                f"tok/s {tokens_per_sec:.0f}"
            )
            running_loss = 0.0
            t0 = time.time()

        # ── Checkpoint ──
        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps:
            ckpt_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), train_cfg.checkpoint_dir
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"model_step{step}.pt")

            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model_cfg,
                    "train_config": train_cfg,
                    "loss": accum_loss,
                },
                ckpt_path,
            )
            print(f"  [Checkpoint] Saved to {ckpt_path}")

            # Quick generation test
            model.eval()
            test_tokens = tokenizer.encode("The Transformer", add_special=True)
            test_tensor = torch.tensor([test_tokens], device=device)
            generated = model.generate(
                test_tensor,
                max_new_tokens=50,
                temperature=0.8,
                top_k=40,
                eos_id=model_cfg.eos_id,
            )
            print(f"  [Sample] {tokenizer.decode(generated)}")
            model.train()

    print("\n[Train] Training complete!")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM-MoE")
    parser.add_argument(
        "--config",
        type=str,
        default="demo",
        choices=["demo", "full"],
        help="Model configuration preset",
    )
    parser.add_argument("--data", type=str, default=None, help="Override data path")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")

    # Web scraping arguments
    parser.add_argument(
        "--scrape", action="store_true",
        help="Enable web scraping for additional training data"
    )
    parser.add_argument(
        "--scrape-urls", nargs="+", default=None,
        help="URLs to scrape for training data"
    )
    parser.add_argument(
        "--scrape-url-file", type=str, default=None,
        help="File with URLs to scrape (one per line)"
    )
    parser.add_argument(
        "--scrape-max-pages", type=int, default=50,
        help="Maximum pages to scrape (default: 50)"
    )
    parser.add_argument(
        "--scrape-depth", type=int, default=2,
        help="Maximum crawl depth (default: 2)"
    )
    parser.add_argument(
        "--scrape-delay", type=float, default=1.0,
        help="Delay between requests in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--no-follow-links", action="store_true",
        help="Do not follow links (only scrape given URLs)"
    )

    args = parser.parse_args()

    # Select config
    model_cfg = DEMO_CONFIG if args.config == "demo" else FULL_CONFIG
    train_cfg = TrainConfig()

    if args.data:
        train_cfg.data_path = args.data
    if args.max_steps:
        train_cfg.max_steps = args.max_steps
    if args.batch_size:
        train_cfg.batch_size = args.batch_size

    # Web scraping config
    if args.scrape or args.scrape_urls or args.scrape_url_file:
        train_cfg.enable_scraping = True
        if args.scrape_urls:
            train_cfg.scrape_urls = args.scrape_urls
        if args.scrape_url_file:
            train_cfg.scrape_url_file = args.scrape_url_file
        train_cfg.scrape_max_pages = args.scrape_max_pages
        train_cfg.scrape_max_depth = args.scrape_depth
        train_cfg.scrape_delay = args.scrape_delay
        train_cfg.scrape_follow_links = not args.no_follow_links

    train(model_cfg, train_cfg)
