from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .muon import CachedPulseMuon, CoherenceMuon, HybridMuon, Muon, PulseMuon

FALLBACK_NAME_FRAGMENTS = (
    "token_embedding",
    "position_embedding",
    "embedding",
    "embed",
    "lm_head",
    "output",
    "unembed",
    "norm",
    "ln_",
    "bias",
)


def split_named_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    fallback_name_fragments: tuple[str, ...] = FALLBACK_NAME_FRAGMENTS,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    matrix_params: list[torch.nn.Parameter] = []
    fallback_params: list[torch.nn.Parameter] = []

    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        lname = name.lower()
        is_excluded = any(fragment in lname for fragment in fallback_name_fragments)
        if param.ndim == 2 and not is_excluded:
            matrix_params.append(param)
        else:
            fallback_params.append(param)
    return matrix_params, fallback_params


def _hybrid_groups(named_parameters, config: dict[str, Any]) -> list[dict[str, Any]]:
    matrix_params, fallback_params = split_named_parameters(named_parameters)
    groups: list[dict[str, Any]] = []
    if matrix_params:
        groups.append({"params": matrix_params, "role": "matrix"})
    if fallback_params:
        groups.append({"params": fallback_params, "role": "fallback"})
    if not groups:
        raise ValueError("No trainable parameters found")
    return groups


def create_optimizer(
    name: str,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    config: dict[str, Any] | None = None,
) -> torch.optim.Optimizer:
    config = dict(config or {})
    key = name.lower()
    params = list(named_parameters)

    if key == "adamw":
        return torch.optim.AdamW(
            [p for _, p in params if p.requires_grad],
            lr=config.get("lr", config.get("fallback_lr", 1e-3)),
            betas=tuple(config.get("betas", (0.9, 0.95))),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.01),
        )
    if key in {"sgd", "nesterov"}:
        return torch.optim.SGD(
            [p for _, p in params if p.requires_grad],
            lr=config.get("lr", 1e-2),
            momentum=config.get("momentum", 0.9),
            nesterov=config.get("nesterov", key == "nesterov"),
            weight_decay=config.get("weight_decay", 0.0),
        )

    groups = _hybrid_groups(params, config)
    common = {
        "matrix_lr": config.get("matrix_lr", config.get("lr", 2e-2)),
        "fallback_lr": config.get("fallback_lr", 1e-3),
        "weight_decay": config.get("weight_decay", 0.01),
        "fallback_weight_decay": config.get("fallback_weight_decay", config.get("weight_decay", 0.01)),
        "momentum": config.get("momentum", 0.95),
        "nesterov": config.get("nesterov", True),
        "ns_steps": config.get("ns_steps", 5),
        "adam_betas": tuple(config.get("adam_betas", (0.9, 0.95))),
        "eps": config.get("eps", 1e-8),
        "rho_min": config.get("rho_min", 0.0),
        "rho_max": config.get("rho_max", 1.0),
        "beta_fast": config.get("beta_fast", 0.90),
        "beta_slow": config.get("beta_slow", 0.99),
        "coherence_gain": config.get("coherence_gain", 6.0),
        "coherence_midpoint": config.get("coherence_midpoint", 0.20),
        "anisotropy_gain": config.get("anisotropy_gain", 3.0),
        "anisotropy_midpoint": config.get("anisotropy_midpoint", 0.15),
        "matrix_update_scale": config.get("matrix_update_scale", 1.0),
        "pulse_interval": config.get("pulse_interval", 4),
        "pulse_warmup_steps": config.get("pulse_warmup_steps", 2),
        "pulse_rho_threshold": config.get("pulse_rho_threshold", 0.85),
    }

    if key == "muon":
        return Muon(groups, **common)
    if key in {"cimuon", "coherence_muon", "coherence-interpolated-muon"}:
        return CoherenceMuon(groups, matrix_mode="full", **common)
    if key in {"pulse_muon", "pmuon", "pulsed_muon"}:
        return PulseMuon(groups, matrix_mode="pulse", **common)
    if key in {"cached_pulse_muon", "cpulse_muon", "cached_pulsed_muon"}:
        return CachedPulseMuon(groups, matrix_mode="cached_pulse", **common)
    if key in {"cimuon_fixed", "fixed_rho"}:
        return HybridMuon(groups, matrix_mode="fixed", fixed_rho=config.get("fixed_rho", 0.5), **common)
    if key in {"cimuon_coherence", "coherence_gate"}:
        return HybridMuon(groups, matrix_mode="coherence", **common)
    if key in {"cimuon_anisotropy", "anisotropy_gate"}:
        return HybridMuon(groups, matrix_mode="anisotropy", **common)
    if key in {"normalized_momentum", "norm_momentum"}:
        return HybridMuon(groups, matrix_mode="normalized", **common)

    raise ValueError(f"Unknown optimizer: {name}")
