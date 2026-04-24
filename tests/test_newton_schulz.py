import torch

from optim_lab.optimizers import (
    matrix_anisotropy,
    newton_schulz_orthogonalize,
    orthogonality_error,
    semi_orthogonal_rms,
)


def test_newton_schulz_preserves_shape_device_dtype() -> None:
    x = torch.randn(16, 8, dtype=torch.float32)
    y = newton_schulz_orthogonalize(x, steps=5)
    assert y.shape == x.shape
    assert y.device == x.device
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_newton_schulz_makes_update_approximately_semi_orthogonal() -> None:
    x = torch.randn(16, 8)
    raw_error = orthogonality_error(x / x.norm())
    y = newton_schulz_orthogonalize(x, steps=5)
    assert orthogonality_error(y) < raw_error
    assert orthogonality_error(y) < 0.75


def test_matrix_anisotropy_detects_energy_concentration() -> None:
    flat = torch.ones(8, 8)
    concentrated = torch.zeros(8, 8)
    concentrated[0, :] = 8.0
    assert matrix_anisotropy(concentrated) > matrix_anisotropy(flat)


def test_semi_orthogonal_rms_matches_shape() -> None:
    assert semi_orthogonal_rms((4, 16)) == 0.25

