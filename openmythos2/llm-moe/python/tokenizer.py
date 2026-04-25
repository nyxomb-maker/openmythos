"""
BPE Tokenizer
=============
Byte Pair Encoding tokenizer implemented from scratch.
Supports training, encoding, decoding, and binary export for the C runtime.
"""

import os
import re
import struct
import json
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional


# Special tokens
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<think>": 4,
    "<step>": 5,
    "<verify>": 6,
    "<conclude>": 7,
    "</think>": 8,
}

# Pre-tokenization regex (similar to GPT-2)
PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.

    Training:
        1. Pre-tokenize text into words
        2. Initialize vocab with byte-level characters
        3. Iteratively merge most frequent pair
        4. Build final vocab up to target size

    Encoding:
        1. Pre-tokenize input
        2. Convert each word to byte sequence
        3. Apply learned merges greedily

    Decoding:
        1. Map token IDs back to byte strings
        2. Concatenate and decode UTF-8
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}          # token_str -> token_id
        self.id_to_token: Dict[int, str] = {}    # token_id -> token_str
        self.merges: List[Tuple[str, str]] = []   # ordered merge rules
        self.merge_ranks: Dict[Tuple[str, str], int] = {}  # merge -> priority
        self.vocab_size: int = 0

    # ── Training ─────────────────────────────────────────────────────────

    def train(self, text: str, vocab_size: int = 4096):
        """
        Train BPE tokenizer on the given text.

        Args:
            text: Training corpus
            vocab_size: Target vocabulary size (including special tokens)
        """
        print(f"[Tokenizer] Training BPE with target vocab_size={vocab_size}")

        # 1. Initialize vocab with special tokens + byte-level chars
        self.vocab = dict(SPECIAL_TOKENS)
        next_id = len(SPECIAL_TOKENS)

        # Add all 256 byte values as base tokens
        for i in range(256):
            byte_token = self._byte_to_token(i)
            if byte_token not in self.vocab:
                self.vocab[byte_token] = next_id
                next_id += 1

        # 2. Pre-tokenize and get word frequencies
        words = PAT.findall(text)
        word_freqs = Counter(words)

        # Convert words to byte-level token sequences
        # word_splits: { tuple_of_byte_tokens: frequency }
        word_splits: Dict[Tuple[str, ...], int] = {}
        for word, freq in word_freqs.items():
            byte_seq = tuple(self._byte_to_token(b) for b in word.encode("utf-8"))
            word_splits[byte_seq] = word_splits.get(byte_seq, 0) + freq

        # 3. Iteratively merge most frequent pairs
        num_merges = vocab_size - next_id
        self.merges = []

        for step in range(num_merges):
            # Count all adjacent pairs
            pair_counts: Dict[Tuple[str, str], int] = Counter()
            for seq, freq in word_splits.items():
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += freq

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            if best_count < 2:
                break

            # Create merged token
            merged = best_pair[0] + best_pair[1]
            self.vocab[merged] = next_id
            self.merges.append(best_pair)
            next_id += 1

            # Apply merge to all word splits
            new_splits = {}
            for seq, freq in word_splits.items():
                new_seq = self._apply_merge(seq, best_pair, merged)
                new_splits[new_seq] = new_splits.get(new_seq, 0) + freq
            word_splits = new_splits

            if (step + 1) % 500 == 0:
                print(
                    f"  Merge {step + 1}/{num_merges}: "
                    f"'{best_pair[0]}' + '{best_pair[1]}' → '{merged}' "
                    f"(count={best_count})"
                )

        # Build reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab_size = len(self.vocab)

        print(f"[Tokenizer] Done. Final vocab_size={self.vocab_size}, merges={len(self.merges)}")

    @staticmethod
    def _apply_merge(
        seq: Tuple[str, ...], pair: Tuple[str, str], merged: str
    ) -> Tuple[str, ...]:
        """Apply a single merge rule to a token sequence."""
        result = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                result.append(merged)
                i += 2
            else:
                result.append(seq[i])
                i += 1
        return tuple(result)

    @staticmethod
    def _byte_to_token(b: int) -> str:
        """Convert a byte value to its token representation."""
        if 33 <= b <= 126 and chr(b) not in ("<", ">"):
            return chr(b)
        return f"<0x{b:02X}>"

    @staticmethod
    def _token_to_bytes(token: str) -> bytes:
        """Convert a token string back to bytes."""
        if token.startswith("<0x") and token.endswith(">"):
            return bytes([int(token[3:5], 16)])
        return token.encode("utf-8")

    # ── Encoding ─────────────────────────────────────────────────────────

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special: If True, prepend BOS and append EOS

        Returns:
            List of token IDs
        """
        tokens = []

        if add_special:
            tokens.append(self.vocab["<bos>"])

        # Check for special tokens in text
        special_pattern = "|".join(
            re.escape(t) for t in SPECIAL_TOKENS if t not in ("<pad>", "<unk>", "<bos>", "<eos>")
        )
        if special_pattern:
            parts = re.split(f"({special_pattern})", text)
        else:
            parts = [text]

        for part in parts:
            if part in self.vocab:
                tokens.append(self.vocab[part])
                continue

            # Pre-tokenize
            words = PAT.findall(part)
            for word in words:
                # Convert to byte tokens
                byte_tokens = [self._byte_to_token(b) for b in word.encode("utf-8")]

                # Apply merges greedily
                word_tokens = self._apply_bpe(byte_tokens)

                for t in word_tokens:
                    tokens.append(self.vocab.get(t, self.vocab["<unk>"]))

        if add_special:
            tokens.append(self.vocab["<eos>"])

        return tokens

    def _apply_bpe(self, tokens: List[str]) -> List[str]:
        """Apply learned BPE merges to a list of tokens."""
        if len(tokens) <= 1:
            return tokens

        while True:
            # Find the merge with the highest priority (lowest rank)
            best_pair = None
            best_rank = float("inf")

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # Apply the merge
            merged = best_pair[0] + best_pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

            if len(tokens) <= 1:
                break

        return tokens

    # ── Decoding ─────────────────────────────────────────────────────────

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special: If True, skip special tokens in output

        Returns:
            Decoded text string
        """
        byte_chunks = []

        for token_id in ids:
            token_str = self.id_to_token.get(token_id, "<unk>")

            # Skip special tokens
            if skip_special and token_str in SPECIAL_TOKENS:
                continue

            # Convert token to bytes
            byte_chunks.append(self._token_to_bytes(token_str))

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    # ── Save / Load (JSON) ───────────────────────────────────────────────

    def save(self, path: str):
        """Save tokenizer to directory (JSON format)."""
        os.makedirs(path, exist_ok=True)

        data = {
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
        }
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[Tokenizer] Saved to {path}")

    def load(self, path: str):
        """Load tokenizer from directory."""
        with open(os.path.join(path, "tokenizer.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = data["vocab"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab_size = len(self.vocab)

        print(f"[Tokenizer] Loaded vocab_size={self.vocab_size}, merges={len(self.merges)}")

    # ── Binary Export (for C runtime) ────────────────────────────────────

    def export_binary(self, filepath: str):
        """
        Export tokenizer in binary format for the C runtime.

        Format:
            vocab_size: uint32
            merges_count: uint32
            For each vocab entry:
                token_len: uint16
                token_bytes: [uint8 × token_len]
                token_id: uint32
            For each merge:
                a_len: uint16
                a_bytes: [uint8 × a_len]
                b_len: uint16
                b_bytes: [uint8 × b_len]
        """
        with open(filepath, "wb") as f:
            # Header
            f.write(struct.pack("<I", self.vocab_size))
            f.write(struct.pack("<I", len(self.merges)))

            # Vocab entries
            for token_str, token_id in sorted(self.vocab.items(), key=lambda x: x[1]):
                token_bytes = token_str.encode("utf-8")
                f.write(struct.pack("<H", len(token_bytes)))
                f.write(token_bytes)
                f.write(struct.pack("<I", token_id))

            # Merge rules
            for a, b in self.merges:
                a_bytes = a.encode("utf-8")
                b_bytes = b.encode("utf-8")
                f.write(struct.pack("<H", len(a_bytes)))
                f.write(a_bytes)
                f.write(struct.pack("<H", len(b_bytes)))
                f.write(b_bytes)

        print(f"[Tokenizer] Exported binary to {filepath}")


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--data", type=str, default="../data/sample.txt")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--output", type=str, default="../tokenizer")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    tok = BPETokenizer()
    tok.train(text, vocab_size=args.vocab_size)
    tok.save(args.output)

    # Quick test
    test_str = "Hello, world! This is a test of the tokenizer."
    encoded = tok.encode(test_str)
    decoded = tok.decode(encoded)
    print(f"\nTest: '{test_str}'")
    print(f"Encoded ({len(encoded)} tokens): {encoded[:20]}...")
    print(f"Decoded: '{decoded}'")

    # Export binary
    tok.export_binary(os.path.join(args.output, "tokenizer.bin"))
