from __future__ import annotations

import math

import torch


def rms(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Root-mean-square with a small floor."""
    return torch.sqrt(torch.mean(x.float().square()) + eps)


def semi_orthogonal_rms(shape: tuple[int, ...]) -> float:
    """RMS of a rectangular semi-orthogonal matrix with this shape."""
    if len(shape) != 2:
        return 1.0
    return 1.0 / math.sqrt(float(max(shape)))


def rms_normalize(
    x: torch.Tensor,
    target_rms: float | torch.Tensor = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    return x * (torch.as_tensor(target_rms, device=x.device, dtype=x.float().dtype) / rms(x, eps))


def newton_schulz_orthogonalize(
    matrix: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> torch.Tensor:
    """Approximate the polar factor of a 2D matrix using Muon's NS5 iteration.

    The calculation is kept in fp32 for broad CPU/MPS compatibility. CUDA users can
    safely adapt this to bf16 if they want closer throughput behavior.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Newton-Schulz orthogonalization expects a 2D tensor, got {matrix.ndim}D")

    original_dtype = matrix.dtype
    x = matrix.float()
    x = x / (x.norm() + eps)

    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    a, b, c = coefficients
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x

    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


def orthogonality_error(matrix: torch.Tensor) -> torch.Tensor:
    """Frobenius error from the nearest semi-orthogonal Gram identity."""
    if matrix.ndim != 2:
        raise ValueError("orthogonality_error expects a 2D tensor")
    x = matrix.float()
    if x.shape[0] >= x.shape[1]:
        gram = x.T @ x
        eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    else:
        gram = x @ x.T
        eye = torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
    return torch.linalg.vector_norm(gram - eye) / math.sqrt(float(eye.numel()))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    af = a.float().flatten()
    bf = b.float().flatten()
    return torch.dot(af, bf) / (torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf) + eps)


def matrix_anisotropy(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Scalar energy-dispersion signal for a matrix update.

    Values near zero mean row/column energy is fairly even. Larger values indicate
    concentrated update energy, where Muon-style spectral balancing is more likely
    to help.
    """
    if matrix.ndim != 2:
        return torch.zeros((), device=matrix.device)
    x2 = matrix.float().square()
    row_energy = x2.mean(dim=1)
    col_energy = x2.mean(dim=0)
    row_cv = row_energy.std(unbiased=False) / (row_energy.mean() + eps)
    col_cv = col_energy.std(unbiased=False) / (col_energy.mean() + eps)
    return torch.log1p(row_cv + col_cv)

