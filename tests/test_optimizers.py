import copy

import torch
import torch.nn as nn

from optim_lab.optimizers import create_optimizer, split_named_parameters


class MixedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(16, 8)
        self.hidden = nn.Linear(8, 8)
        self.ln_1 = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.scalar = nn.Parameter(torch.zeros(()))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(idx).mean(dim=1)
        x = self.ln_1(self.hidden(x))
        return self.lm_head(x).square().mean() + self.scalar.square()


def contains_identity(params: list[torch.nn.Parameter], target: torch.nn.Parameter) -> bool:
    return any(param is target for param in params)


def test_split_named_parameters_uses_muon_only_for_hidden_matrices() -> None:
    model = MixedModel()
    matrix, fallback = split_named_parameters(model.named_parameters())
    assert contains_identity(matrix, model.hidden.weight)
    assert contains_identity(fallback, model.token_embedding.weight)
    assert contains_identity(fallback, model.lm_head.weight)
    assert contains_identity(fallback, model.hidden.bias)
    assert contains_identity(fallback, model.scalar)


def test_coherence_muon_steps_mixed_parameter_model() -> None:
    torch.manual_seed(0)
    model = MixedModel()
    optimizer = create_optimizer(
        "cimuon",
        model.named_parameters(),
        {"matrix_lr": 0.02, "fallback_lr": 0.001, "rho_min": 0.05, "rho_max": 0.95},
    )
    before = copy.deepcopy([p.detach().clone() for p in model.parameters()])
    loss = model(torch.randint(0, 16, (4, 6)))
    loss.backward()
    optimizer.step()
    after = [p.detach().clone() for p in model.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
    assert optimizer.last_stats["matrix_count"] == 1.0
    assert 0.05 <= optimizer.last_stats["rho"] <= 0.95


def test_fixed_rho_reports_requested_value() -> None:
    torch.manual_seed(0)
    model = MixedModel()
    optimizer = create_optimizer(
        "cimuon_fixed",
        model.named_parameters(),
        {"matrix_lr": 0.02, "fallback_lr": 0.001, "fixed_rho": 0.25},
    )
    loss = model(torch.randint(0, 16, (4, 6)))
    loss.backward()
    optimizer.step()
    assert abs(optimizer.last_stats["rho"] - 0.25) < 1e-6


def test_muon_reports_pure_orthogonalization_even_with_candidate_clamps() -> None:
    torch.manual_seed(0)
    model = MixedModel()
    optimizer = create_optimizer(
        "muon",
        model.named_parameters(),
        {"matrix_lr": 0.02, "fallback_lr": 0.001, "rho_min": 0.05, "rho_max": 0.95},
    )
    loss = model(torch.randint(0, 16, (4, 6)))
    loss.backward()
    optimizer.step()
    assert abs(optimizer.last_stats["rho"] - 1.0) < 1e-6


def test_pulse_muon_can_skip_newton_schulz_after_warmup() -> None:
    torch.manual_seed(0)
    model = MixedModel()
    optimizer = create_optimizer(
        "pulse_muon",
        model.named_parameters(),
        {
            "matrix_lr": 0.02,
            "fallback_lr": 0.001,
            "pulse_warmup_steps": 1,
            "pulse_interval": 100,
            "pulse_rho_threshold": 1.1,
        },
    )
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randint(0, 16, (4, 6)))
        loss.backward()
        optimizer.step()
    assert optimizer.last_stats["ns_fraction"] == 0.0
    assert optimizer.last_stats["desired_rho"] > 0.0


def test_cached_pulse_muon_reuses_orthogonal_direction_after_warmup() -> None:
    torch.manual_seed(0)
    model = MixedModel()
    optimizer = create_optimizer(
        "cached_pulse_muon",
        model.named_parameters(),
        {
            "matrix_lr": 0.02,
            "fallback_lr": 0.001,
            "pulse_warmup_steps": 1,
            "pulse_interval": 100,
            "pulse_rho_threshold": 1.1,
        },
    )
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randint(0, 16, (4, 6)))
        loss.backward()
        optimizer.step()
    assert optimizer.last_stats["ns_fraction"] == 0.0
    assert optimizer.last_stats["rho"] > 0.0
