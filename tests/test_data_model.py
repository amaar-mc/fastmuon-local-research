import torch

from optim_lab.data import TokenStream, generate_synthetic_tokens
from optim_lab.models import GPTConfig, TinyGPT


def test_synthetic_tokens_are_deterministic() -> None:
    a = generate_synthetic_tokens(256, vocab_size=64, seed=123)
    b = generate_synthetic_tokens(256, vocab_size=64, seed=123)
    c = generate_synthetic_tokens(256, vocab_size=64, seed=124)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_token_stream_batch_shapes() -> None:
    stream = TokenStream.from_config(
        {"vocab_size": 64, "block_size": 16, "n_tokens": 1000, "val_fraction": 0.2},
        seed=0,
    )
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    x, y = stream.get_batch("train", 4, "cpu", g)
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert x.dtype == torch.long


def test_tiny_gpt_forward_loss() -> None:
    config = GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = TinyGPT(config)
    x = torch.randint(0, 32, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 32)
    assert loss is not None
    assert torch.isfinite(loss)

