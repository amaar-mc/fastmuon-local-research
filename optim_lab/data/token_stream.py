from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def generate_synthetic_tokens(
    n_tokens: int,
    vocab_size: int,
    seed: int,
    mode: str = "algorithmic",
) -> torch.Tensor:
    """Generate a deterministic token stream with learnable structure."""
    if n_tokens < 2:
        raise ValueError("n_tokens must be at least 2")
    if vocab_size < 8:
        raise ValueError("vocab_size must be at least 8")

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    if mode == "markov":
        transitions = torch.randint(0, vocab_size, (vocab_size, 4), generator=g)
        tokens = torch.empty(n_tokens, dtype=torch.long)
        tokens[0] = torch.randint(0, vocab_size, (), generator=g)
        for i in range(1, n_tokens):
            branch = int((tokens[i - 1] + i) % 4)
            noise = torch.randint(0, 7, (), generator=g)
            tokens[i] = (transitions[tokens[i - 1], branch] + noise) % vocab_size
        return tokens

    if mode != "algorithmic":
        raise ValueError(f"Unknown synthetic mode: {mode}")

    t = torch.arange(n_tokens, dtype=torch.long)
    periods = torch.tensor([7, 17, 31, 53], dtype=torch.long)
    base = (
        3 * (t % periods[0])
        + 5 * ((t // periods[1]) % periods[1])
        + 11 * ((t // periods[2]) % periods[2])
        + 13 * ((t * t + 19 * t) % periods[3])
    )
    noise_mask = torch.rand(n_tokens, generator=g) < 0.03
    noise = torch.randint(0, vocab_size, (n_tokens,), generator=g)
    tokens = base % vocab_size
    tokens = torch.where(noise_mask, noise, tokens)
    return tokens.long()


def load_text_tokens(path: str | Path, vocab_size: int) -> torch.Tensor:
    """Byte-level tokenization, folded into vocab_size for small experiments."""
    data = Path(path).read_bytes()
    if len(data) < 2:
        raise ValueError(f"Text file is too small: {path}")
    values = torch.tensor(list(data), dtype=torch.long)
    return values % vocab_size


@dataclass
class TokenStream:
    train: torch.Tensor
    val: torch.Tensor
    block_size: int

    @classmethod
    def from_config(cls, config: dict, seed: int) -> "TokenStream":
        vocab_size = int(config.get("vocab_size", 256))
        block_size = int(config.get("block_size", 64))
        n_tokens = int(config.get("n_tokens", 200_000))
        val_fraction = float(config.get("val_fraction", 0.1))
        if config.get("text_path"):
            tokens = load_text_tokens(config["text_path"], vocab_size)
        else:
            tokens = generate_synthetic_tokens(
                n_tokens=n_tokens,
                vocab_size=vocab_size,
                seed=seed,
                mode=config.get("mode", "algorithmic"),
            )
        split = max(block_size + 2, int(tokens.numel() * (1.0 - val_fraction)))
        split = min(split, tokens.numel() - block_size - 2)
        if split <= block_size:
            raise ValueError("Not enough tokens for train/val split")
        return cls(train=tokens[:split], val=tokens[split:], block_size=block_size)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        if data.numel() <= self.block_size + 1:
            raise ValueError(f"{split} split too small for block_size={self.block_size}")
        ix = torch.randint(0, data.numel() - self.block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

