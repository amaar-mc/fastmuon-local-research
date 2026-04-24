from __future__ import annotations

import math
from typing import Any

import torch

from .utils import (
    cosine_similarity,
    matrix_anisotropy,
    newton_schulz_orthogonalize,
    orthogonality_error,
    rms,
    rms_normalize,
    semi_orthogonal_rms,
)


class HybridMuon(torch.optim.Optimizer):
    """Hybrid optimizer with Muon-style matrix updates and AdamW fallback.

    Matrix parameter groups use Nesterov momentum followed by either plain Muon,
    normalized momentum, fixed interpolation, or coherence/anisotropy gated
    interpolation. Fallback groups use AdamW, matching common Muon practice for
    embeddings, heads, biases, LayerNorm, scalars, and vectors.
    """

    def __init__(
        self,
        params,
        *,
        matrix_lr: float = 2e-2,
        fallback_lr: float = 1e-3,
        weight_decay: float = 0.01,
        fallback_weight_decay: float | None = None,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        matrix_mode: str = "muon",
        fixed_rho: float | None = None,
        rho_min: float = 0.0,
        rho_max: float = 1.0,
        beta_fast: float = 0.90,
        beta_slow: float = 0.99,
        coherence_gain: float = 6.0,
        coherence_midpoint: float = 0.20,
        anisotropy_gain: float = 3.0,
        anisotropy_midpoint: float = 0.15,
        matrix_update_scale: float = 1.0,
        pulse_interval: int = 4,
        pulse_warmup_steps: int = 2,
        pulse_rho_threshold: float = 0.85,
    ) -> None:
        if matrix_mode not in {
            "muon",
            "normalized",
            "fixed",
            "coherence",
            "anisotropy",
            "full",
            "pulse",
            "cached_pulse",
        }:
            raise ValueError(f"Unknown matrix_mode: {matrix_mode}")
        if not 0.0 <= rho_min <= rho_max <= 1.0:
            raise ValueError("Expected 0 <= rho_min <= rho_max <= 1")
        if fixed_rho is not None and not 0.0 <= fixed_rho <= 1.0:
            raise ValueError("fixed_rho must be in [0, 1]")

        fallback_weight_decay = weight_decay if fallback_weight_decay is None else fallback_weight_decay
        defaults = dict(
            matrix_lr=matrix_lr,
            fallback_lr=fallback_lr,
            weight_decay=weight_decay,
            fallback_weight_decay=fallback_weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adam_betas=adam_betas,
            eps=eps,
            matrix_mode=matrix_mode,
            fixed_rho=fixed_rho,
            rho_min=rho_min,
            rho_max=rho_max,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            coherence_gain=coherence_gain,
            coherence_midpoint=coherence_midpoint,
            anisotropy_gain=anisotropy_gain,
            anisotropy_midpoint=anisotropy_midpoint,
            matrix_update_scale=matrix_update_scale,
            pulse_interval=pulse_interval,
            pulse_warmup_steps=pulse_warmup_steps,
            pulse_rho_threshold=pulse_rho_threshold,
        )
        super().__init__(params, defaults)
        self.last_stats: dict[str, float] = {}

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        stats: dict[str, float] = {
            "matrix_count": 0.0,
            "fallback_count": 0.0,
            "rho_sum": 0.0,
            "coherence_sum": 0.0,
            "anisotropy_sum": 0.0,
            "update_rms_sum": 0.0,
            "update_to_weight_sum": 0.0,
            "ns_error_sum": 0.0,
            "ns_error_count": 0.0,
            "ns_call_count": 0.0,
            "desired_rho_sum": 0.0,
        }

        for group in self.param_groups:
            role = group.get("role", "fallback")
            if role == "matrix":
                self._step_matrix_group(group, stats)
            else:
                self._step_adamw_group(group, stats)

        matrix_count = max(stats["matrix_count"], 1.0)
        fallback_count = max(stats["fallback_count"], 1.0)
        self.last_stats = {
            "matrix_count": stats["matrix_count"],
            "fallback_count": stats["fallback_count"],
            "rho": stats["rho_sum"] / matrix_count,
            "coherence": stats["coherence_sum"] / matrix_count,
            "anisotropy": stats["anisotropy_sum"] / matrix_count,
            "update_rms": stats["update_rms_sum"] / matrix_count,
            "update_to_weight": stats["update_to_weight_sum"] / matrix_count,
            "ns_error": stats["ns_error_sum"] / matrix_count,
            "ns_error_on_calls": stats["ns_error_sum"] / max(stats["ns_error_count"], 1.0),
            "ns_fraction": stats["ns_call_count"] / matrix_count,
            "desired_rho": stats["desired_rho_sum"] / matrix_count,
            "fallback_steps": stats["fallback_count"] / fallback_count,
        }
        return loss

    def _matrix_rho(self, group: dict[str, Any], coherence: torch.Tensor, anisotropy: torch.Tensor) -> torch.Tensor:
        mode = group["matrix_mode"]
        if mode == "muon":
            return torch.ones_like(coherence)
        elif mode == "normalized":
            return torch.zeros_like(coherence)
        elif mode == "fixed":
            return torch.as_tensor(group["fixed_rho"], device=coherence.device, dtype=coherence.dtype)
        else:
            coherence_gate = torch.sigmoid(group["coherence_gain"] * (coherence - group["coherence_midpoint"]))
            anisotropy_gate = torch.sigmoid(group["anisotropy_gain"] * (anisotropy - group["anisotropy_midpoint"]))
            if mode == "coherence":
                rho = coherence_gate
            elif mode == "anisotropy":
                rho = anisotropy_gate
            elif mode in {"full", "pulse", "cached_pulse"}:
                rho = coherence_gate * anisotropy_gate
            else:
                raise ValueError(f"Unknown matrix mode: {mode}")
        return torch.clamp(rho, group["rho_min"], group["rho_max"])

    def _should_pulse_refresh(self, group: dict[str, Any], state: dict[str, Any], rho: torch.Tensor) -> bool:
        state["pulse_step"] += 1
        state["pulse_age"] += 1
        if state["pulse_step"] <= group["pulse_warmup_steps"]:
            return True
        if state["pulse_age"] >= group["pulse_interval"]:
            return True
        if float(rho.detach().cpu()) >= group["pulse_rho_threshold"]:
            return True
        return False

    def _step_matrix_group(self, group: dict[str, Any], stats: dict[str, float]) -> None:
        lr = group["matrix_lr"]
        wd = group["weight_decay"]
        momentum = group["momentum"]
        beta_fast = group["beta_fast"]
        beta_slow = group["beta_slow"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("HybridMuon does not support sparse gradients")
            if p.ndim != 2:
                raise RuntimeError("Matrix group received a non-2D parameter")

            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)
                state["m_fast"] = torch.zeros_like(p)
                state["m_slow"] = torch.zeros_like(p)
                state["cached_ortho"] = torch.zeros_like(p)
                state["pulse_step"] = 0
                state["pulse_age"] = 0

            if wd:
                p.mul_(1.0 - lr * wd)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)
            update_source = grad.add(buf, alpha=momentum) if group["nesterov"] else buf

            m_fast = state["m_fast"]
            m_slow = state["m_slow"]
            m_fast.mul_(beta_fast).add_(grad, alpha=1.0 - beta_fast)
            m_slow.mul_(beta_slow).add_(grad, alpha=1.0 - beta_slow)

            coherence = cosine_similarity(m_fast, m_slow).clamp(-1.0, 1.0)
            anisotropy = matrix_anisotropy(update_source)
            desired_rho = self._matrix_rho(group, coherence, anisotropy)
            rho = desired_rho
            target_rms = semi_orthogonal_rms(tuple(p.shape)) * group["matrix_update_scale"]
            normalized = rms_normalize(update_source, target_rms=target_rms, eps=group["eps"])

            ns_called = False
            ns_err = None
            if group["matrix_mode"] in {"pulse", "cached_pulse"}:
                if self._should_pulse_refresh(group, state, desired_rho):
                    ortho = newton_schulz_orthogonalize(update_source, steps=group["ns_steps"], eps=group["eps"])
                    state["cached_ortho"].copy_(ortho)
                    state["pulse_age"] = 0
                    ns_called = True
                    ns_err = orthogonality_error(ortho)
                else:
                    ortho = state["cached_ortho"]
                    if group["matrix_mode"] == "pulse":
                        rho = torch.zeros_like(desired_rho)
            elif float(desired_rho.detach().cpu()) <= 1e-8:
                ortho = torch.zeros_like(normalized)
                rho = torch.zeros_like(desired_rho)
            else:
                ortho = newton_schulz_orthogonalize(update_source, steps=group["ns_steps"], eps=group["eps"])
                ns_called = True
                ns_err = orthogonality_error(ortho)

            mixed = (1.0 - rho) * normalized + rho * ortho
            update = rms_normalize(mixed, target_rms=target_rms, eps=group["eps"])

            p.add_(update, alpha=-lr)

            stats["matrix_count"] += 1.0
            stats["rho_sum"] += float(rho.detach().cpu())
            stats["desired_rho_sum"] += float(desired_rho.detach().cpu())
            stats["coherence_sum"] += float(coherence.detach().cpu())
            stats["anisotropy_sum"] += float(anisotropy.detach().cpu())
            stats["update_rms_sum"] += float(rms(update).detach().cpu())
            stats["update_to_weight_sum"] += float((lr * update.norm() / (p.norm() + group["eps"])).detach().cpu())
            if ns_called and ns_err is not None:
                stats["ns_call_count"] += 1.0
                stats["ns_error_count"] += 1.0
                stats["ns_error_sum"] += float(ns_err.detach().cpu())

    def _step_adamw_group(self, group: dict[str, Any], stats: dict[str, float]) -> None:
        lr = group["fallback_lr"]
        wd = group["fallback_weight_decay"]
        beta1, beta2 = group["adam_betas"]
        eps = group["eps"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("HybridMuon fallback does not support sparse gradients")

            state = self.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros((), device=p.device, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            if wd:
                p.mul_(1.0 - lr * wd)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"].add_(1.0)
            step = float(state["step"].detach().cpu())

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            denom = exp_avg_sq.sqrt().add_(eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)

            stats["fallback_count"] += 1.0


class Muon(HybridMuon):
    def __init__(self, params, **kwargs) -> None:
        kwargs["matrix_mode"] = "muon"
        super().__init__(params, **kwargs)


class CoherenceMuon(HybridMuon):
    def __init__(self, params, **kwargs) -> None:
        kwargs.setdefault("matrix_mode", "full")
        super().__init__(params, **kwargs)


class PulseMuon(HybridMuon):
    def __init__(self, params, **kwargs) -> None:
        kwargs.setdefault("matrix_mode", "pulse")
        super().__init__(params, **kwargs)


class CachedPulseMuon(HybridMuon):
    def __init__(self, params, **kwargs) -> None:
        kwargs.setdefault("matrix_mode", "cached_pulse")
        super().__init__(params, **kwargs)
