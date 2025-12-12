"""
Polarity Engine

Implements Level 1 of PNA: dual generative/radiative agents with
rhythmic balance inspired by Walter Russell's cosmological principles
(The Universal One, 1926).

Mathematical Foundation:
    Ψ = |dG/dt + dR/dt|²

Where G represents compression/generative processes and R represents
expansion/radiative processes. The engine maintains balanced interchange
through minimizing the polarity penalty Ψ.

Maps to:
    - Exploration (R) vs Exploitation (G) in RL
    - Encoding vs Decoding in autoencoders
    - Compression vs Expansion in information theory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class PolarityEngine(nn.Module):
    """
    Polarity Engine implementing rhythmic balanced interchange.

    Dual-agent system with:
        - Generative Agent (G): Compression, exploitation, inward flow
        - Radiative Agent (R): Expansion, exploration, outward flow

    Maintains balance through polarity penalty minimization.

    Attributes:
        hidden_dim: Dimension of hidden representations
        latent_dim: Dimension of latent space
        lambda_penalty: Weight for polarity penalty
        balance_threshold: Threshold for considering system balanced
        octave_cycles: Number of octave cycles for rhythmic patterns
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        lambda_penalty: float = 1.0,
        balance_threshold: float = 0.1,
        octave_cycles: int = 8,
        enable_rhythmic_modulation: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lambda_penalty = lambda_penalty
        self.balance_threshold = balance_threshold
        self.octave_cycles = octave_cycles
        self.enable_rhythmic_modulation = enable_rhythmic_modulation

        # Generative Agent (Compression/Exploitation)
        self.generative_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        self.generative_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Radiative Agent (Expansion/Exploration)
        self.radiative_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.radiative_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Rate trackers (dG/dt and dR/dt)
        self.register_buffer('gen_rate', torch.tensor(0.0))
        self.register_buffer('rad_rate', torch.tensor(0.0))
        self.register_buffer('gen_rate_history', torch.zeros(100))
        self.register_buffer('rad_rate_history', torch.zeros(100))
        self.register_buffer('step_counter', torch.tensor(0))

        # Octave modulation parameters (Russell's octave cycles)
        if enable_rhythmic_modulation:
            self.octave_freqs = nn.Parameter(
                torch.logspace(0, np.log2(octave_cycles), octave_cycles, base=2.0)
            )
            self.octave_phases = nn.Parameter(torch.zeros(octave_cycles))
        else:
            self.register_buffer('octave_freqs', torch.ones(octave_cycles))
            self.register_buffer('octave_phases', torch.zeros(octave_cycles))

    def compute_polarity_penalty(
        self,
        gen_rate: Optional[torch.Tensor] = None,
        rad_rate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute polarity penalty: Ψ = |dG/dt + dR/dt|²

        Perfect balance: dG/dt = -dR/dt (equal and opposite rates)
        Results in Ψ = 0

        Args:
            gen_rate: Generative rate dG/dt (optional, uses self.gen_rate if None)
            rad_rate: Radiative rate dR/dt (optional, uses self.rad_rate if None)

        Returns:
            Polarity penalty scalar
        """
        if gen_rate is None:
            gen_rate = self.gen_rate
        if rad_rate is None:
            rad_rate = self.rad_rate

        # Ψ = |dG/dt + dR/dt|²
        penalty = torch.abs(gen_rate + rad_rate) ** 2

        return penalty

    def compute_rates(
        self,
        x: torch.Tensor,
        prev_gen_loss: Optional[torch.Tensor] = None,
        prev_rad_loss: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute instantaneous rates dG/dt and dR/dt.

        In discrete time: dG/dt ≈ ΔG = L_gen(t) - L_gen(t-1)

        Args:
            x: Input tensor
            prev_gen_loss: Previous generative loss
            prev_rad_loss: Previous radiative loss

        Returns:
            Dictionary with rates and losses
        """
        # Forward through both agents
        gen_latent = self.generative_encoder(x)
        gen_recon = self.generative_decoder(gen_latent)
        gen_loss = F.mse_loss(gen_recon, x)

        rad_latent = self.radiative_encoder(x)
        rad_recon = self.radiative_decoder(rad_latent)
        rad_loss = F.mse_loss(rad_recon, x)

        # Compute rates (if previous losses available)
        if prev_gen_loss is not None:
            gen_rate = gen_loss - prev_gen_loss
        else:
            gen_rate = torch.tensor(0.0, device=x.device)

        if prev_rad_loss is not None:
            rad_rate = rad_loss - prev_rad_loss
        else:
            rad_rate = torch.tensor(0.0, device=x.device)

        return {
            'gen_rate': gen_rate,
            'rad_rate': rad_rate,
            'gen_loss': gen_loss,
            'rad_loss': rad_loss,
            'gen_latent': gen_latent,
            'rad_latent': rad_latent,
            'gen_recon': gen_recon,
            'rad_recon': rad_recon,
        }

    def octave_modulation(self, step: int) -> torch.Tensor:
        """
        Compute octave-based rhythmic modulation.

        Based on Russell's principle of octave cycles in nature.
        Creates harmonic patterns at different timescales.

        Args:
            step: Current time step

        Returns:
            Modulation factor [0, 1]
        """
        if not self.enable_rhythmic_modulation:
            return torch.tensor(1.0)

        # Sum of sine waves at octave frequencies
        t = step / 100.0  # Normalize time
        modulation = torch.sum(
            torch.sin(2 * np.pi * self.octave_freqs * t + self.octave_phases)
        ) / len(self.octave_freqs)

        # Normalize to [0, 1]
        modulation = (modulation + 1.0) / 2.0

        return modulation

    def forward(
        self,
        x: torch.Tensor,
        mode: str = 'balanced',
        return_metrics: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through polarity engine.

        Args:
            x: Input tensor [batch_size, input_dim]
            mode: Operating mode - 'balanced', 'generative', 'radiative', 'adaptive'
            return_metrics: Whether to return detailed metrics

        Returns:
            - Generative output
            - Radiative output
            - Metrics dictionary (optional)
        """
        batch_size = x.shape[0]

        # Compute rates and outputs
        rate_dict = self.compute_rates(x)

        # Update internal rate trackers
        with torch.no_grad():
            self.gen_rate = rate_dict['gen_rate'].detach()
            self.rad_rate = rate_dict['rad_rate'].detach()

            # Update history
            idx = int(self.step_counter % 100)
            self.gen_rate_history[idx] = self.gen_rate
            self.rad_rate_history[idx] = self.rad_rate
            self.step_counter += 1

        # Compute polarity penalty
        penalty = self.compute_polarity_penalty()

        # Apply octave modulation
        octave_mod = self.octave_modulation(int(self.step_counter))

        # Mode-dependent combination
        if mode == 'balanced':
            # Equal weighting
            output = 0.5 * rate_dict['gen_recon'] + 0.5 * rate_dict['rad_recon']
            latent = 0.5 * rate_dict['gen_latent'] + 0.5 * rate_dict['rad_latent']
        elif mode == 'generative':
            # Favor compression/exploitation
            output = rate_dict['gen_recon']
            latent = rate_dict['gen_latent']
        elif mode == 'radiative':
            # Favor expansion/exploration
            output = rate_dict['rad_recon']
            latent = rate_dict['rad_latent']
        elif mode == 'adaptive':
            # Weight by octave modulation
            w = octave_mod
            output = w * rate_dict['gen_recon'] + (1 - w) * rate_dict['rad_recon']
            latent = w * rate_dict['gen_latent'] + (1 - w) * rate_dict['rad_latent']
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not return_metrics:
            return rate_dict['gen_recon'], rate_dict['rad_recon'], None

        # Compile metrics
        metrics = {
            'polarity_penalty': penalty,
            'gen_rate': self.gen_rate,
            'rad_rate': self.rad_rate,
            'gen_loss': rate_dict['gen_loss'],
            'rad_loss': rate_dict['rad_loss'],
            'octave_modulation': octave_mod,
            'is_balanced': penalty < self.balance_threshold,
            'gen_latent': rate_dict['gen_latent'],
            'rad_latent': rate_dict['rad_latent'],
            'combined_output': output,
            'combined_latent': latent,
        }

        return rate_dict['gen_recon'], rate_dict['rad_recon'], metrics

    def update_with_balance(
        self,
        x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        mode: str = 'balanced'
    ) -> Dict[str, float]:
        """
        Training step with polarity penalty.

        Loss = L_task + λ * Ψ

        Args:
            x: Input batch
            optimizer: Optimizer instance
            mode: Operating mode

        Returns:
            Dictionary of loss components
        """
        optimizer.zero_grad()

        # Forward pass
        gen_out, rad_out, metrics = self.forward(x, mode=mode, return_metrics=True)

        # Task loss (reconstruction)
        task_loss = metrics['gen_loss'] + metrics['rad_loss']

        # Polarity penalty
        penalty_loss = self.lambda_penalty * metrics['polarity_penalty']

        # Total loss
        total_loss = task_loss + penalty_loss

        # Backward
        total_loss.backward()
        optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'penalty_loss': penalty_loss.item(),
            'gen_loss': metrics['gen_loss'].item(),
            'rad_loss': metrics['rad_loss'].item(),
            'polarity_penalty': metrics['polarity_penalty'].item(),
            'is_balanced': metrics['is_balanced'].item(),
        }

    def get_balance_metrics(self) -> Dict[str, float]:
        """
        Get current balance state metrics.

        Returns:
            Dictionary of balance-related metrics
        """
        with torch.no_grad():
            penalty = self.compute_polarity_penalty()

            # Rate statistics from history
            gen_rate_mean = self.gen_rate_history.mean().item()
            gen_rate_std = self.gen_rate_history.std().item()
            rad_rate_mean = self.rad_rate_history.mean().item()
            rad_rate_std = self.rad_rate_history.std().item()

        return {
            'current_penalty': penalty.item(),
            'is_balanced': penalty.item() < self.balance_threshold,
            'gen_rate': self.gen_rate.item(),
            'rad_rate': self.rad_rate.item(),
            'gen_rate_mean': gen_rate_mean,
            'gen_rate_std': gen_rate_std,
            'rad_rate_mean': rad_rate_mean,
            'rad_rate_std': rad_rate_std,
            'rate_correlation': np.corrcoef(
                self.gen_rate_history.cpu().numpy(),
                self.rad_rate_history.cpu().numpy()
            )[0, 1] if self.step_counter > 10 else 0.0,
        }


class AdaptivePolarityEngine(PolarityEngine):
    """
    Adaptive Polarity Engine with automatic balance adjustment.

    Learns optimal balance point through meta-learning.
    """

    def __init__(self, *args, meta_lr: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)

        self.meta_lr = meta_lr

        # Learnable balance weight
        self.balance_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        mode: str = 'adaptive',
        return_metrics: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward with adaptive weighting."""
        gen_out, rad_out, metrics = super().forward(x, mode='balanced', return_metrics=True)

        if mode == 'adaptive':
            # Adaptive combination based on learned weight
            w = torch.sigmoid(self.balance_weight)
            output = w * gen_out + (1 - w) * rad_out

            if metrics is not None:
                metrics['balance_weight'] = w
                metrics['combined_output'] = output

        return gen_out, rad_out, metrics


if __name__ == "__main__":
    # Example usage
    print("Testing Polarity Engine...")

    # Initialize
    engine = PolarityEngine(input_dim=64, hidden_dim=128, latent_dim=32)

    # Create synthetic data
    x = torch.randn(8, 64)

    # Forward pass
    gen_out, rad_out, metrics = engine(x, mode='balanced')

    print(f"\nPolarity Penalty: {metrics['polarity_penalty'].item():.6f}")
    print(f"Generative Rate: {metrics['gen_rate'].item():.6f}")
    print(f"Radiative Rate: {metrics['rad_rate'].item():.6f}")
    print(f"Is Balanced: {metrics['is_balanced'].item()}")
    print(f"Octave Modulation: {metrics['octave_modulation'].item():.4f}")

    # Test adaptive mode
    print("\n\nTesting Adaptive Polarity Engine...")
    adaptive_engine = AdaptivePolarityEngine(input_dim=64)
    gen_out, rad_out, metrics = adaptive_engine(x, mode='adaptive')
    print(f"Balance Weight: {metrics['balance_weight'].item():.4f}")
