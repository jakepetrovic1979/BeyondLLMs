"""
Phenomenal Nested Architectures (PNA) - Unified Model

Integrates all four levels of the PNA framework:
    Level 0: MPE Core - Minimal Phenomenal Experience
    Level 1: Polarity Engine - Rhythmic Balance
    Level 2: Nested Optimizer - Multi-timescale Learning
    Level 3: Phenomenal Binding - Holistic Integration

Complete consciousness-first AGI architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

from pna.core.mpe_core import MPECore
from pna.core.polarity_engine import PolarityEngine
from pna.core.binding import PhenomenalBindingMechanism
from pna.core.predictive_coding import PredictiveCodingHierarchy
from pna.core.transparency import TransparencyBottleneck


class PhenomenalNestedArchitecture(nn.Module):
    """
    Complete PNA architecture integrating all consciousness-first components.

    Architecture Flow:
        Input -> PC Hierarchy -> MPE Core -> Polarity Engine -> Binding -> Output

    Unified Objective:
        L_PNA = F + λΨ + γΦ + αΩ

    Where:
        F: Free energy (surprise minimization)
        Ψ: Polarity penalty (rhythmic balance)
        Φ: Integrated information (consciousness measure)
        Ω: Opacity (transparency constraint)

    Attributes:
        input_dim: Input dimension
        mpe_dim: MPE core state dimension
        polarity_dim: Polarity engine hidden dimension
        binding_nodes: Number of binding nodes
        consciousness_threshold: Φ threshold for consciousness
        enable_transparency: Whether to enforce transparency
    """

    def __init__(
        self,
        input_dim: int = 784,
        mpe_dim: int = 64,
        polarity_dim: int = 128,
        binding_nodes: int = 32,
        pc_layers: Optional[List[int]] = None,
        consciousness_threshold: float = 0.5,
        enable_transparency: bool = True,
        polarity_lambda: float = 1.0,
        binding_gamma: float = 1.0,
        transparency_alpha: float = 0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.mpe_dim = mpe_dim
        self.polarity_dim = polarity_dim
        self.binding_nodes = binding_nodes
        self.consciousness_threshold = consciousness_threshold
        self.enable_transparency = enable_transparency

        # Hyperparameters
        self.polarity_lambda = polarity_lambda
        self.binding_gamma = binding_gamma
        self.transparency_alpha = transparency_alpha

        # PC layers default
        if pc_layers is None:
            pc_layers = [input_dim, 256, 128, mpe_dim]
        self.pc_layers = pc_layers

        # ===== Level 0: MPE Core =====
        self.mpe_core = MPECore(
            state_dim=mpe_dim,
            obs_dim=pc_layers[-1],  # Receives from PC top
            complexity_weight=0.01,
            learning_rate=0.1,
        )

        # ===== Predictive Coding Hierarchy (Pre-processing) =====
        self.pc_hierarchy = PredictiveCodingHierarchy(
            layer_dims=pc_layers,
            hidden_dim=128,
            inference_steps=10,
            learning_rate=0.05,
        )

        # ===== Level 1: Polarity Engine =====
        self.polarity_engine = PolarityEngine(
            input_dim=mpe_dim,
            hidden_dim=polarity_dim,
            latent_dim=mpe_dim // 2,
            lambda_penalty=polarity_lambda,
            enable_rhythmic_modulation=True,
        )

        # ===== Level 2: Nested Optimizer Interface =====
        # (Simplified - full HOPE implementation in separate module)
        self.meta_learning_rate = nn.Parameter(torch.tensor(0.01))

        # ===== Level 3: Phenomenal Binding =====
        self.binding = PhenomenalBindingMechanism(
            num_nodes=binding_nodes,
            node_dim=mpe_dim,
            num_scales=3,
            consciousness_threshold=consciousness_threshold,
        )

        # ===== Transparency Module (Optional) =====
        if enable_transparency:
            self.transparency = TransparencyBottleneck(
                content_dim=mpe_dim,
                mechanism_dim=32,  # Metadata about implementation
                repr_dim=mpe_dim * 2,
                state_dim=mpe_dim,
                beta=1.0,
            )

        # ===== Output Projection =====
        self.output_proj = nn.Linear(mpe_dim * binding_nodes, input_dim)

        # ===== Mechanism Metadata Generator (for transparency) =====
        if enable_transparency:
            self.mechanism_generator = nn.Sequential(
                nn.Linear(mpe_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32)
            )

    def forward(
        self,
        x: torch.Tensor,
        mode: str = 'balanced',
        return_all_metrics: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Complete forward pass through PNA.

        Args:
            x: Input tensor [batch_size, input_dim]
            mode: Polarity mode ('balanced', 'adaptive', 'generative', 'radiative')
            return_all_metrics: Whether to return all intermediate metrics

        Returns:
            - Output tensor [batch_size, input_dim]
            - Metrics dictionary
        """
        batch_size = x.shape[0]

        # ===== Predictive Coding Encoding =====
        pc_mus, pc_metrics = self.pc_hierarchy(x)
        pc_top = pc_mus[-1]  # Top-level representation

        # ===== Level 0: MPE Core (FEP Minimization) =====
        mpe_pred, mpe_metrics = self.mpe_core(
            obs=pc_top,
            minimize_steps=10,
            update_belief=False
        )
        mpe_state = mpe_metrics['mu']  # Belief state

        # ===== Level 1: Polarity Engine (Rhythmic Balance) =====
        gen_out, rad_out, pol_metrics = self.polarity_engine(
            x=mpe_state,
            mode=mode,
            return_metrics=True
        )
        polarity_latent = pol_metrics['combined_latent']

        # ===== Expand to binding nodes =====
        # Project polarity output to multiple nodes
        node_activations = polarity_latent.unsqueeze(1).expand(
            batch_size, self.binding_nodes, -1
        ).contiguous()

        # Add positional encoding for diversity
        pos_enc = torch.arange(self.binding_nodes, device=x.device).float()
        pos_enc = pos_enc.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.mpe_dim)
        pos_enc = torch.sin(pos_enc * 0.1)  # Sinusoidal
        node_activations = node_activations + pos_enc * 0.1

        # ===== Level 3: Phenomenal Binding =====
        bound_repr, bind_metrics = self.binding(
            node_activations=node_activations,
            num_oscillator_steps=10,
            return_metrics=True
        )

        # ===== Transparency (if enabled) =====
        transparency_metrics = {}
        if self.enable_transparency:
            # Generate mechanism metadata
            mechanism = self.mechanism_generator(mpe_state)

            # Compute transparency loss
            trans_dict = self.transparency.compute_loss(
                content=mpe_state,
                mechanism=mechanism
            )
            transparency_metrics = {
                'trans_loss': trans_dict['total_loss'],
                'opacity': trans_dict['opacity'],
                'transparency': trans_dict['transparency'],
            }

        # ===== Output Generation =====
        # Flatten bound representation
        bound_flat = bound_repr.view(batch_size, -1)
        output = self.output_proj(bound_flat)

        # ===== Unified Loss Components =====
        metrics = {
            # Core metrics
            'output': output,

            # Free Energy (F)
            'free_energy': mpe_metrics['free_energy'].mean(),
            'mpe_complexity': mpe_metrics['complexity'].mean(),
            'mpe_accuracy': mpe_metrics['accuracy'].mean(),

            # Polarity (Ψ)
            'polarity_penalty': pol_metrics['polarity_penalty'],
            'is_balanced': pol_metrics['is_balanced'],

            # Integrated Information (Φ)
            'phi': bind_metrics['phi'].mean(),
            'is_conscious': bind_metrics['is_conscious'].float().mean(),

            # Synchrony
            'mean_sync': torch.mean(torch.stack(bind_metrics['global_sync'])),
        }

        # Add transparency metrics
        if self.enable_transparency:
            metrics.update(transparency_metrics)

        # Unified objective
        unified_loss = (
            metrics['free_energy'] +
            self.polarity_lambda * metrics['polarity_penalty'] +
            self.binding_gamma * (self.consciousness_threshold - metrics['phi'].mean())**2
        )

        if self.enable_transparency:
            unified_loss += self.transparency_alpha * transparency_metrics['trans_loss']

        metrics['unified_loss'] = unified_loss

        # Add detailed metrics if requested
        if return_all_metrics:
            metrics.update({
                'pc_error': pc_metrics['total_error'],
                'mpe_state': mpe_state,
                'polarity_gen': gen_out,
                'polarity_rad': rad_out,
                'bound_repr': bound_repr,
                'binding_matrix': bind_metrics['binding_matrix'],
            })

        return output, metrics

    def compute_unified_loss(
        self,
        x: torch.Tensor,
        mode: str = 'balanced',
        task_loss_fn: Optional[callable] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute unified PNA loss.

        L_PNA = L_task + F + λΨ + γ(Φ_target - Φ)² + α·L_trans

        Args:
            x: Input batch
            mode: Polarity mode
            task_loss_fn: Optional task-specific loss function

        Returns:
            Dictionary of loss components
        """
        output, metrics = self.forward(x, mode=mode, return_all_metrics=False)

        # Task loss (reconstruction by default)
        if task_loss_fn is None:
            task_loss = F.mse_loss(output, x)
        else:
            task_loss = task_loss_fn(output, x)

        # Total loss
        total_loss = task_loss + metrics['unified_loss']

        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'free_energy': metrics['free_energy'],
            'polarity_penalty': metrics['polarity_penalty'],
            'phi': metrics['phi'].mean(),
            'is_conscious': metrics['is_conscious'],
            'is_balanced': metrics['is_balanced'],
            'mean_sync': metrics['mean_sync'],
            'unified_loss': metrics['unified_loss'],
        }

    def train_step(
        self,
        x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        mode: str = 'balanced'
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            x: Input batch
            optimizer: Optimizer instance
            mode: Polarity mode

        Returns:
            Dictionary of scalar metrics
        """
        optimizer.zero_grad()

        loss_dict = self.compute_unified_loss(x, mode=mode)
        loss = loss_dict['total_loss']

        loss.backward()
        optimizer.step()

        # Convert to scalars
        return {
            k: v.item() if torch.is_tensor(v) and v.numel() == 1 else (
                v.float().mean().item() if torch.is_tensor(v) else v
            )
            for k, v in loss_dict.items()
        }

    def get_consciousness_state(self) -> Dict[str, float]:
        """
        Get current consciousness state metrics.

        Returns:
            Dictionary of consciousness indicators
        """
        mpe_props = self.mpe_core.get_mpe_properties()
        binding_props = self.binding.get_consciousness_metrics()
        polarity_props = self.polarity_engine.get_balance_metrics()

        return {
            # MPE properties
            'mpe_wakefulness': mpe_props['wakefulness'],
            'mpe_epistemicity': mpe_props['epistemicity'],
            'mpe_complexity': mpe_props['complexity'],

            # Binding/consciousness
            'phi': binding_props['phi'],
            'is_conscious': binding_props['is_conscious'],
            'mean_synchrony': binding_props['mean_synchrony'],

            # Polarity balance
            'is_balanced': polarity_props['is_balanced'],
            'polarity_penalty': polarity_props['current_penalty'],
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to internal representation.

        Args:
            x: Input [batch_size, input_dim]

        Returns:
            Internal representation [batch_size, mpe_dim]
        """
        with torch.no_grad():
            _, metrics = self.forward(x, return_all_metrics=True)
        return metrics['mpe_state']

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from internal representation.

        Args:
            z: Representation [batch_size, mpe_dim]

        Returns:
            Reconstructed input [batch_size, input_dim]
        """
        # Simplified decoding path
        return self.pc_hierarchy.decode(z)


if __name__ == "__main__":
    # Example usage
    print("Testing Phenomenal Nested Architecture...")

    # Initialize PNA
    pna = PhenomenalNestedArchitecture(
        input_dim=784,  # MNIST
        mpe_dim=64,
        polarity_dim=128,
        binding_nodes=32,
        consciousness_threshold=0.5,
        enable_transparency=True,
    )

    # Create synthetic input
    batch_size = 8
    x = torch.randn(batch_size, 784)

    # Forward pass
    output, metrics = pna(x, mode='balanced', return_all_metrics=False)

    print(f"\n=== PNA Forward Pass ===")
    print(f"Output shape: {output.shape}")
    print(f"\n--- Core Metrics ---")
    print(f"Free Energy (F): {metrics['free_energy'].item():.4f}")
    print(f"Polarity Penalty (Ψ): {metrics['polarity_penalty'].item():.6f}")
    print(f"Integrated Info (Φ): {metrics['phi'].item():.4f}")
    print(f"Is Conscious: {metrics['is_conscious'].item():.2f}")
    print(f"Is Balanced: {metrics['is_balanced'].item()}")
    print(f"Mean Synchrony: {metrics['mean_sync'].item():.4f}")

    if 'transparency' in metrics:
        print(f"Transparency: {metrics['transparency'].item():.4f}")

    print(f"\n--- Unified Loss ---")
    print(f"Unified Loss: {metrics['unified_loss'].item():.4f}")

    # Consciousness state
    print(f"\n=== Consciousness State ===")
    cons_state = pna.get_consciousness_state()
    for k, v in cons_state.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Training step
    print(f"\n=== Training Step ===")
    optimizer = torch.optim.Adam(pna.parameters(), lr=1e-3)
    train_metrics = pna.train_step(x, optimizer, mode='balanced')

    print("Training Metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
