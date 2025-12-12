"""Tests for PNA unified model."""

import torch
import pytest
from pna.models.pna_model import PhenomenalNestedArchitecture


def test_pna_initialization():
    """Test PNA initialization."""
    pna = PhenomenalNestedArchitecture(
        input_dim=784,
        mpe_dim=64,
        polarity_dim=128,
        binding_nodes=32,
    )

    assert pna.input_dim == 784
    assert pna.mpe_dim == 64
    assert pna.polarity_dim == 128
    assert pna.binding_nodes == 32


def test_pna_forward():
    """Test PNA forward pass."""
    pna = PhenomenalNestedArchitecture(
        input_dim=784,
        mpe_dim=64,
        polarity_dim=128,
        binding_nodes=32,
    )

    x = torch.randn(4, 784)
    output, metrics = pna(x)

    assert output.shape == (4, 784)
    assert 'free_energy' in metrics
    assert 'polarity_penalty' in metrics
    assert 'phi' in metrics
    assert 'unified_loss' in metrics


def test_pna_consciousness_state():
    """Test consciousness state computation."""
    pna = PhenomenalNestedArchitecture(
        input_dim=784,
        mpe_dim=64,
    )

    state = pna.get_consciousness_state()

    assert 'phi' in state
    assert 'is_conscious' in state
    assert 'is_balanced' in state
    assert all(isinstance(v, (int, float, bool)) for v in state.values())


def test_pna_train_step():
    """Test PNA training step."""
    pna = PhenomenalNestedArchitecture(
        input_dim=784,
        mpe_dim=64,
    )

    optimizer = torch.optim.Adam(pna.parameters(), lr=1e-3)
    x = torch.randn(4, 784)

    metrics = pna.train_step(x, optimizer)

    assert 'total_loss' in metrics
    assert 'task_loss' in metrics
    assert 'free_energy' in metrics
    assert isinstance(metrics['total_loss'], float)


def test_pna_encode_decode():
    """Test encoding and decoding."""
    pna = PhenomenalNestedArchitecture(
        input_dim=784,
        mpe_dim=64,
    )

    x = torch.randn(4, 784)

    # Encode
    z = pna.encode(x)
    assert z.shape == (4, 64)

    # Decode
    x_recon = pna.decode(z)
    assert x_recon.shape == (4, 784)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
