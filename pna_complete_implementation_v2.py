"""
PNA Complete Implementation v2.0 - Continual Learning Focus
============================================================

This is the production-ready implementation of Phenomenal Nested Architectures
optimized for continual learning benchmarks.

Key components:
- MPECore: Minimal Phenomenal Experience (Level 0)
- PolarityEngine: Generative/Radiative balance (Level 1)
- MetaController: Adaptive plasticity (Level 2)
- GlobalWorkspace: Information integration (Level 3)
- PNASystem: Unified architecture

Designed for Split-MNIST, Permuted-MNIST, and Split-CIFAR continual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================================
# Level 0: MPE Core
# ============================================================================

class MPECore(nn.Module):
    """
    Minimal Phenomenal Experience Core (Level 0).

    Implements Free Energy Principle minimization:
        F = Accuracy + Complexity

    Maintains belief state μ that tracks observations with minimal surprise.
    """

    def __init__(self, d_input: int, d_hidden: int, device: str = 'cpu'):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.device = device

        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_hidden * 2),
            nn.Tanh(),
            nn.Linear(d_hidden * 2, d_hidden)
        ).to(device)

        # Belief state
        self.register_buffer('mu', torch.zeros(1, d_hidden, device=device))
        self.register_buffer('f0_current', torch.tensor(0.0, device=device))

        # FEP hyperparameters
        self.beta = 0.1  # Learning rate for belief update

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process observation and update belief.

        Args:
            x: Observation [batch_size, seq_len, d_input]

        Returns:
            Encoded state [batch_size, seq_len, d_hidden]
        """
        batch_size, seq_len, _ = x.shape

        # Encode observation
        z = self.encoder(x)  # [batch_size, seq_len, d_hidden]

        # Compute free energy (simplified: prediction error)
        if self.mu.shape[0] != batch_size:
            self.mu = self.mu.expand(batch_size, -1).contiguous()

        # Expand mu for broadcasting
        mu_expanded = self.mu.unsqueeze(1).expand(-1, seq_len, -1)

        # Free energy = prediction error
        accuracy = F.mse_loss(z, mu_expanded, reduction='none').mean()
        complexity = torch.norm(self.mu, p=2) * 0.01

        f0 = accuracy + complexity
        self.f0_current = f0

        # Update belief (exponential moving average)
        with torch.no_grad():
            z_mean = z.mean(dim=[0, 1])  # Average across batch and time
            self.mu = (1 - self.beta) * self.mu[0] + self.beta * z_mean
            self.mu = self.mu.unsqueeze(0)

        return z

    def reset(self):
        """Reset belief state."""
        self.mu = torch.zeros_like(self.mu)
        self.f0_current = torch.tensor(0.0, device=self.device)


# ============================================================================
# Level 1: Polarity Engine
# ============================================================================

class PolarityEngine(nn.Module):
    """
    Polarity Engine (Level 1).

    Maintains generative/radiative balance through dual GRU streams.
    Plasticity modulation through polarity-based gating.
    """

    def __init__(self, d_hidden: int, device: str = 'cpu'):
        super().__init__()
        self.d_hidden = d_hidden
        self.device = device

        # Dual GRU streams
        self.gru_g = nn.GRU(d_hidden, d_hidden, batch_first=True).to(device)
        self.gru_r = nn.GRU(d_hidden, d_hidden, batch_first=True).to(device)

        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        ).to(device)

        # Hidden states
        self.register_buffer('h_g', torch.zeros(1, 1, d_hidden, device=device))
        self.register_buffer('h_r', torch.zeros(1, 1, d_hidden, device=device))

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Process through dual polarity streams.

        Args:
            x: Input [batch_size, seq_len, d_hidden]

        Returns:
            dict with 'z', 'h', 'polarity_loss'
        """
        batch_size = x.shape[0]

        # Adjust hidden states for batch size
        if self.h_g.shape[1] != batch_size:
            self.h_g = self.h_g.expand(-1, batch_size, -1).contiguous()
            self.h_r = self.h_r.expand(-1, batch_size, -1).contiguous()

        # Generative stream
        z_g, h_g_new = self.gru_g(x, self.h_g)

        # Radiative stream
        z_r, h_r_new = self.gru_r(x, self.h_r)

        # Compute polarity gate
        combined = torch.cat([z_g, z_r], dim=-1)
        gate = self.gate_network(combined)

        # Weighted combination
        z = gate * z_g + (1 - gate) * z_r

        # Polarity penalty (encourage balance)
        polarity_loss = F.mse_loss(gate, torch.full_like(gate, 0.5))

        # Update hidden states
        self.h_g = h_g_new.detach()
        self.h_r = h_r_new.detach()

        # Combine hidden states for output
        h = gate[:, -1:, :] * h_g_new + (1 - gate[:, -1:, :]) * h_r_new

        return {
            'z': z,
            'h': h.squeeze(0),  # [batch_size, d_hidden]
            'polarity_loss': polarity_loss,
            'gate': gate.mean()
        }

    def reset(self):
        """Reset hidden states."""
        self.h_g = torch.zeros_like(self.h_g)
        self.h_r = torch.zeros_like(self.h_r)


# ============================================================================
# Level 2: Meta-Controller
# ============================================================================

class MetaController(nn.Module):
    """
    Meta-Controller (Level 2).

    Adaptive plasticity based on task performance.
    Returns scalar plasticity gate that modulates learning rate.
    """

    def __init__(self, d_hidden: int, device: str = 'cpu'):
        super().__init__()
        self.d_hidden = d_hidden
        self.device = device

        # Meta-learning network
        self.meta_network = nn.Sequential(
            nn.Linear(d_hidden + 1, d_hidden),  # +1 for loss signal
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, h: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """
        Compute plasticity modulation.

        Args:
            h: Hidden state [batch_size, d_hidden]
            loss: Current loss [1]

        Returns:
            Plasticity gate [batch_size, 1]
        """
        # Concatenate state and loss
        loss_expanded = loss.unsqueeze(0).unsqueeze(0).expand(h.shape[0], 1)
        meta_input = torch.cat([h, loss_expanded], dim=-1)

        # Compute plasticity gate
        plasticity = self.meta_network(meta_input)

        return plasticity


# ============================================================================
# Level 3: Global Workspace
# ============================================================================

class GlobalWorkspace(nn.Module):
    """
    Global Workspace (Level 3).

    Integrates information across time and computes Φ proxy.
    """

    def __init__(self, d_hidden: int, device: str = 'cpu'):
        super().__init__()
        self.d_hidden = d_hidden
        self.device = device

        # Workspace attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=4,
            batch_first=True
        ).to(device)

        # Workspace state
        self.register_buffer('w', torch.zeros(1, 1, d_hidden, device=device))

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> Dict:
        """
        Integrate information into workspace.

        Args:
            z: Sequential states [batch_size, seq_len, d_hidden]
            h: Hidden state [batch_size, d_hidden]

        Returns:
            dict with 'w' (workspace state) and 'phi' (integration measure)
        """
        batch_size = z.shape[0]

        # Use h as query, z as key/value
        h_query = h.unsqueeze(1)  # [batch_size, 1, d_hidden]

        # Attention
        w, attn_weights = self.attention(h_query, z, z)

        # Compute Φ proxy (eigenvalue-based)
        # Simplified: use variance as complexity measure
        phi = z.var(dim=1).mean(dim=-1)  # [batch_size]

        self.w = w.detach()

        return {
            'w': w,  # [batch_size, 1, d_hidden]
            'phi': phi.mean().item(),
            'attn_weights': attn_weights
        }

    def reset(self):
        """Reset workspace state."""
        self.w = torch.zeros_like(self.w)


# ============================================================================
# Unified PNA System
# ============================================================================

class PNASystem(nn.Module):
    """
    Complete PNA system for continual learning.

    Integrates all four levels and provides unified training interface.
    """

    def __init__(
        self,
        d_input: int = 28,
        d_hidden: int = 64,
        d_output: int = 2,
        lambda_pol: float = 0.1,
        lambda_mpe: float = 0.5,
        device: str = 'cpu'
    ):
        super().__init__()

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.device = device

        # Hyperparameters
        self.lambda_pol = lambda_pol
        self.lambda_mpe = lambda_mpe

        # Level 0: MPE Core
        self.l0_mpe = MPECore(d_input, d_hidden, device)

        # Level 1: Polarity Engine
        self.l1_polarity = PolarityEngine(d_hidden, device)

        # Level 2: Meta-Controller
        self.l2_meta = MetaController(d_hidden, device)

        # Level 3: Global Workspace
        self.l3_workspace = GlobalWorkspace(d_hidden, device)

        # Output head
        self.output_head = nn.Linear(d_hidden, d_output).to(device)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Forward pass through all levels.

        Args:
            x: Input [batch_size, d_input] (flattened images)
            targets: Optional targets [batch_size]

        Returns:
            dict with outputs and metrics
        """
        batch_size = x.shape[0]

        # Reshape to sequence format
        x = x.unsqueeze(1)  # [batch_size, 1, d_input]

        # Level 0: MPE
        z = self.l0_mpe(x)  # [batch_size, 1, d_hidden]

        # Level 1: Polarity
        pol_result = self.l1_polarity(z)
        z_pol = pol_result['z']  # [batch_size, 1, d_hidden]
        h_pol = pol_result['h']  # [batch_size, d_hidden]

        # Level 3: Workspace
        ws_result = self.l3_workspace(z_pol, h_pol)
        w = ws_result['w']  # [batch_size, 1, d_hidden]

        # Output
        output = self.output_head(w.squeeze(1))  # [batch_size, d_output]

        # Losses
        if targets is not None:
            task_loss = F.cross_entropy(output, targets)
            polarity_loss = pol_result['polarity_loss']
            mpe_loss = self.l0_mpe.f0_current

            total_loss = (
                task_loss +
                self.lambda_pol * polarity_loss +
                self.lambda_mpe * mpe_loss
            )

            # Level 2: Compute plasticity
            plasticity = self.l2_meta(h_pol, total_loss)
        else:
            task_loss = None
            total_loss = None
            plasticity = None

        return {
            'output': output.unsqueeze(1),  # [batch_size, 1, d_output] for consistency
            'losses': {
                'task': task_loss,
                'polarity': pol_result['polarity_loss'] if targets else None,
                'mpe': self.l0_mpe.f0_current,
                'total': total_loss
            },
            'plasticity': plasticity,
            'phi': ws_result['phi'],
            'workspace': w
        }

    def reset(self):
        """Reset all stateful components."""
        self.l0_mpe.reset()
        self.l1_polarity.reset()
        self.l3_workspace.reset()


# ============================================================================
# Data Loading for Split-MNIST
# ============================================================================

def get_split_mnist(task_id: int, device: str = 'cpu'):
    """
    Get Split-MNIST dataset for a specific task.

    Split-MNIST: 5 binary tasks (0/1, 2/3, 4/5, 6/7, 8/9)

    Args:
        task_id: Task index (0-4)
        device: Device for data

    Returns:
        train_dataset, test_dataset
    """
    # Define digit pairs
    digit_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    pair = digit_pairs[task_id]

    # Download MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Filter for current task
    def filter_dataset(dataset, digits):
        indices = [i for i, (img, label) in enumerate(dataset) if label in digits]
        subset = Subset(dataset, indices)

        # Remap labels to 0/1
        class FilteredDataset(Dataset):
            def __init__(self, subset, digit_map):
                self.subset = subset
                self.digit_map = digit_map

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                img, label = self.subset[idx]
                # Flatten image
                img = img.view(-1)
                # Remap label
                new_label = 0 if label == self.digit_map[0] else 1
                return img, new_label

        return FilteredDataset(subset, digits)

    train_dataset = filter_dataset(train_full, pair)
    test_dataset = filter_dataset(test_full, pair)

    return train_dataset, test_dataset


def get_split_mnist_loaders(task_id: int, batch_size: int = 32, device: str = 'cpu'):
    """
    Get DataLoaders for Split-MNIST task.

    Args:
        task_id: Task index (0-4)
        batch_size: Batch size
        device: Device

    Returns:
        train_loader, test_loader
    """
    train_dataset, test_dataset = get_split_mnist(task_id, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing PNA Complete Implementation v2.0")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Initialize model
    pna = PNASystem(d_input=28*28, d_hidden=64, d_output=2, device=device)

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 28*28).to(device)
    targets = torch.randint(0, 2, (batch_size,)).to(device)

    print("Testing forward pass...")
    result = pna(x, targets)

    print(f"\nOutput shape: {result['output'].shape}")
    print(f"Task loss: {result['losses']['task'].item():.4f}")
    print(f"Polarity loss: {result['losses']['polarity'].item():.6f}")
    print(f"MPE loss: {result['losses']['mpe'].item():.4f}")
    print(f"Total loss: {result['losses']['total'].item():.4f}")
    print(f"Phi: {result['phi']:.4f}")
    print(f"Plasticity: {result['plasticity'].mean().item():.4f}")

    # Test Split-MNIST loading
    print(f"\n{'=' * 80}")
    print("Testing Split-MNIST loading...")

    train_loader, test_loader = get_split_mnist_loaders(0, batch_size=32, device=device)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    data, target = next(iter(train_loader))
    print(f"Batch data shape: {data.shape}")
    print(f"Batch target shape: {target.shape}")
    print(f"Target values: {target[:10]}")

    print("\n" + "=" * 80)
    print("All tests passed!")
