"""Tests for MPE Core."""

import torch
import pytest
from pna.core.mpe_core import MPECore


def test_mpe_initialization():
    """Test MPE Core initialization."""
    mpe = MPECore(state_dim=32, obs_dim=64)
    assert mpe.state_dim == 32
    assert mpe.obs_dim == 64
    assert mpe.mu.shape == (32,)


def test_mpe_forward():
    """Test MPE forward pass."""
    mpe = MPECore(state_dim=32, obs_dim=64)
    obs = torch.randn(4, 64)

    predictions, metrics = mpe(obs, minimize_steps=10)

    assert predictions.shape == (4, 64)
    assert 'free_energy' in metrics
    assert 'complexity' in metrics
    assert 'accuracy' in metrics
    assert 'prediction_error' in metrics


def test_mpe_free_energy_minimization():
    """Test that free energy decreases during minimization."""
    mpe = MPECore(state_dim=32, obs_dim=64)
    obs = torch.randn(4, 64)

    # Get initial free energy
    mu_init = torch.randn(4, 32)
    prec_init = torch.ones(4, 32)
    metrics_init = mpe.compute_free_energy(obs, mu_init, prec_init)
    fe_init = metrics_init['free_energy'].mean()

    # Minimize
    mu_final, prec_final, metrics_final = mpe.minimize_free_energy(obs, iterations=20)
    fe_final = metrics_final['free_energy'].mean()

    # Should decrease (or at least not increase significantly)
    assert fe_final <= fe_init + 0.1  # Allow small tolerance


def test_mpe_properties():
    """Test MPE property computation."""
    mpe = MPECore(state_dim=32, obs_dim=64)
    props = mpe.get_mpe_properties()

    assert 'wakefulness' in props
    assert 'complexity' in props
    assert 'epistemicity' in props
    assert all(isinstance(v, (int, float)) for v in props.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
