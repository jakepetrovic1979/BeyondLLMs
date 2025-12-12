"""
Representational Transparency and Opacity

Implements the transparency constraint for phenomenal representations,
ensuring that phenomenological content is independent of implementation
mechanism (substrate independence).

Mathematical Foundation:
    Ω(r, s) = 1 - I(M(r); S(s)) / H(M(r))

Where:
    - r: Representation
    - s: Phenomenal state
    - M(r): Mechanism/substrate information
    - S(s): Phenomenal content
    - I: Mutual information
    - H: Entropy

Goal: Ω → 1 (perfect transparency)
    - Phenomenology reveals content, not mechanism
    - Substrate independence
    - Implementation invariance

Implementation via Information Bottleneck:
    L = I(S; C) - β I(S; M)

    Maximize content fidelity, minimize mechanism visibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class MutualInformationEstimator(nn.Module):
    """
    Neural estimator for mutual information.

    Uses MINE (Mutual Information Neural Estimation) or similar approaches.

    I(X; Y) = E_p[log(T(x,y))] - log(E_q[exp(T(x,y))])

    where:
        - T: Statistics network
        - p: Joint distribution
        - q: Product of marginals
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        # Statistics network T(x, y)
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Estimate I(X; Y).

        Args:
            x: Samples from X [batch_size, x_dim]
            y: Samples from Y [batch_size, y_dim]

        Returns:
            MI estimate (scalar)
        """
        batch_size = x.shape[0]

        # Joint samples
        joint = torch.cat([x, y], dim=-1)
        joint_scores = self.net(joint)

        # Marginal samples (shuffle y)
        y_shuffle = y[torch.randperm(batch_size)]
        marginal = torch.cat([x, y_shuffle], dim=-1)
        marginal_scores = self.net(marginal)

        # MINE bound
        mi = torch.mean(joint_scores) - torch.log(torch.mean(torch.exp(marginal_scores)) + 1e-8)

        return mi


class TransparencyBottleneck(nn.Module):
    """
    Information bottleneck for transparency.

    Architecture:
        Content (C) -> Representation (r) -> Phenomenal State (s)
                              ↑
                         Mechanism (M)

    Objective:
        Maximize I(S; C) - content fidelity
        Minimize I(S; M) - mechanism visibility

    This ensures phenomenal states reveal content, not substrate.

    Attributes:
        content_dim: Dimension of content
        mechanism_dim: Dimension of mechanism metadata
        repr_dim: Dimension of representation
        state_dim: Dimension of phenomenal state
        beta: Weight for mechanism independence (higher = more transparent)
    """

    def __init__(
        self,
        content_dim: int = 64,
        mechanism_dim: int = 32,
        repr_dim: int = 128,
        state_dim: int = 64,
        beta: float = 1.0,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.content_dim = content_dim
        self.mechanism_dim = mechanism_dim
        self.repr_dim = repr_dim
        self.state_dim = state_dim
        self.beta = beta

        # Encoder: (Content, Mechanism) -> Representation
        self.encoder = nn.Sequential(
            nn.Linear(content_dim + mechanism_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, repr_dim)
        )

        # Phenomenal mapping: Representation -> State
        self.phenomenal_map = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Content reconstruction: State -> Content
        self.content_decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, content_dim)
        )

        # MI estimators
        self.mi_content = MutualInformationEstimator(
            x_dim=state_dim,
            y_dim=content_dim,
            hidden_dim=64
        )

        self.mi_mechanism = MutualInformationEstimator(
            x_dim=state_dim,
            y_dim=mechanism_dim,
            hidden_dim=64
        )

    def forward(
        self,
        content: torch.Tensor,
        mechanism: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through transparency bottleneck.

        Args:
            content: Content information [batch_size, content_dim]
            mechanism: Mechanism metadata [batch_size, mechanism_dim]

        Returns:
            - Representation [batch_size, repr_dim]
            - Phenomenal state [batch_size, state_dim]
            - Reconstructed content [batch_size, content_dim]
        """
        # Encode with both content and mechanism
        combined = torch.cat([content, mechanism], dim=-1)
        representation = self.encoder(combined)

        # Map to phenomenal state (should reflect content, not mechanism)
        state = self.phenomenal_map(representation)

        # Reconstruct content
        content_recon = self.content_decoder(state)

        return representation, state, content_recon

    def compute_opacity(
        self,
        state: torch.Tensor,
        mechanism: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute representational opacity.

        Ω = 1 - I(M; S) / H(M)

        Approximation: Use MI estimate and assume H(M) ≈ log(mechanism_dim)

        Args:
            state: Phenomenal states [batch_size, state_dim]
            mechanism: Mechanism info [batch_size, mechanism_dim]

        Returns:
            Opacity scalar ∈ [0, 1] (higher = more transparent)
        """
        # Estimate I(M; S)
        mi_mech_state = self.mi_mechanism(state, mechanism)

        # Approximate entropy (upper bound)
        h_mechanism = np.log(self.mechanism_dim)

        # Opacity
        opacity = 1.0 - (mi_mech_state / h_mechanism).clamp(0.0, 1.0)

        return opacity

    def compute_loss(
        self,
        content: torch.Tensor,
        mechanism: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute transparency loss.

        L = -I(S; C) + β I(S; M) + reconstruction_error

        Args:
            content: Content information
            mechanism: Mechanism metadata

        Returns:
            Dictionary of loss components
        """
        # Forward
        representation, state, content_recon = self.forward(content, mechanism)

        # Reconstruction fidelity
        recon_loss = F.mse_loss(content_recon, content)

        # MI with content (maximize)
        mi_content_state = self.mi_content(state, content)

        # MI with mechanism (minimize)
        mi_mech_state = self.mi_mechanism(state, mechanism)

        # Total loss
        total_loss = -mi_content_state + self.beta * mi_mech_state + recon_loss

        # Opacity metric
        opacity = self.compute_opacity(state, mechanism)

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'mi_content': mi_content_state,
            'mi_mechanism': mi_mech_state,
            'opacity': opacity,
            'transparency': 1.0 - opacity,  # For clarity
            'state': state,
            'representation': representation,
        }

    def train_step(
        self,
        content: torch.Tensor,
        mechanism: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            content: Content batch
            mechanism: Mechanism batch
            optimizer: Optimizer

        Returns:
            Dictionary of metrics
        """
        optimizer.zero_grad()

        loss_dict = self.compute_loss(content, mechanism)
        loss = loss_dict['total_loss']

        loss.backward()
        optimizer.step()

        return {
            k: v.item() if torch.is_tensor(v) and v.numel() == 1 else v
            for k, v in loss_dict.items()
            if k not in ['state', 'representation']
        }

    def test_substrate_independence(
        self,
        content: torch.Tensor,
        mechanism1: torch.Tensor,
        mechanism2: torch.Tensor
    ) -> Dict[str, float]:
        """
        Test substrate independence.

        Compute phenomenal states with different mechanisms,
        verify they're similar (content determines state, not substrate).

        Args:
            content: Shared content
            mechanism1: First substrate/mechanism
            mechanism2: Second substrate/mechanism

        Returns:
            Independence metrics
        """
        with torch.no_grad():
            _, state1, _ = self.forward(content, mechanism1)
            _, state2, _ = self.forward(content, mechanism2)

            # States should be similar
            state_similarity = F.cosine_similarity(state1, state2, dim=-1).mean()
            state_distance = F.mse_loss(state1, state2)

        return {
            'state_similarity': state_similarity.item(),
            'state_distance': state_distance.item(),
            'substrate_independent': state_similarity.item() > 0.9,
        }


class AdaptiveTransparency(TransparencyBottleneck):
    """
    Adaptive transparency with learnable beta.

    Automatically balances content fidelity and mechanism independence.
    """

    def __init__(self, *args, init_beta: float = 1.0, **kwargs):
        super().__init__(*args, beta=init_beta, **kwargs)

        # Learnable beta
        self.log_beta = nn.Parameter(torch.log(torch.tensor(init_beta)))

    @property
    def beta(self) -> torch.Tensor:
        """Current beta value."""
        return torch.exp(self.log_beta).clamp(min=0.01, max=10.0)

    def compute_loss(
        self,
        content: torch.Tensor,
        mechanism: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute loss with adaptive beta."""
        # Get base losses
        loss_dict = super().compute_loss(content, mechanism)

        # Update with current beta
        loss_dict['beta'] = self.beta
        loss_dict['total_loss'] = (
            -loss_dict['mi_content'] +
            self.beta * loss_dict['mi_mechanism'] +
            loss_dict['recon_loss']
        )

        return loss_dict


class TransparencyEvaluator:
    """
    Evaluator for transparency properties.

    Tests:
        1. Substrate independence
        2. Content fidelity
        3. Mechanism invisibility
        4. Invariance to implementation changes
    """

    def __init__(self, model: TransparencyBottleneck):
        self.model = model

    def evaluate(
        self,
        content: torch.Tensor,
        mechanism: torch.Tensor,
        num_substrate_tests: int = 5
    ) -> Dict[str, float]:
        """
        Comprehensive transparency evaluation.

        Args:
            content: Content samples
            mechanism: Primary mechanism
            num_substrate_tests: Number of substrate variants to test

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        with torch.no_grad():
            # Base forward
            _, state, content_recon = self.model(content, mechanism)

            # Content fidelity
            content_fidelity = -F.mse_loss(content_recon, content).item()

            # Opacity
            opacity = self.model.compute_opacity(state, mechanism).item()

            # Test multiple substrates
            substrate_similarities = []
            for _ in range(num_substrate_tests):
                alt_mechanism = torch.randn_like(mechanism)
                metrics = self.model.test_substrate_independence(
                    content, mechanism, alt_mechanism
                )
                substrate_similarities.append(metrics['state_similarity'])

            mean_substrate_sim = np.mean(substrate_similarities)
            std_substrate_sim = np.std(substrate_similarities)

        return {
            'content_fidelity': content_fidelity,
            'opacity': opacity,
            'transparency': 1.0 - opacity,
            'substrate_similarity_mean': mean_substrate_sim,
            'substrate_similarity_std': std_substrate_sim,
            'substrate_independent': mean_substrate_sim > 0.85,
        }


if __name__ == "__main__":
    # Example usage
    print("Testing Transparency Bottleneck...")

    # Initialize
    transparency = TransparencyBottleneck(
        content_dim=64,
        mechanism_dim=32,
        repr_dim=128,
        state_dim=64,
        beta=2.0
    )

    # Create synthetic data
    batch_size = 16
    content = torch.randn(batch_size, 64)
    mechanism = torch.randn(batch_size, 32)

    # Compute loss
    loss_dict = transparency.compute_loss(content, mechanism)

    print(f"\nTotal Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Reconstruction Loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"MI(S; C): {loss_dict['mi_content'].item():.4f}")
    print(f"MI(S; M): {loss_dict['mi_mechanism'].item():.4f}")
    print(f"Opacity (Ω): {loss_dict['opacity'].item():.4f}")
    print(f"Transparency: {loss_dict['transparency'].item():.4f}")

    # Test substrate independence
    mechanism2 = torch.randn(batch_size, 32)
    independence = transparency.test_substrate_independence(content, mechanism, mechanism2)
    print(f"\nSubstrate Independence:")
    print(f"  State Similarity: {independence['state_similarity']:.4f}")
    print(f"  State Distance: {independence['state_distance']:.6f}")
    print(f"  Independent: {independence['substrate_independent']}")

    # Comprehensive evaluation
    print("\n\nComprehensive Evaluation...")
    evaluator = TransparencyEvaluator(transparency)
    eval_metrics = evaluator.evaluate(content, mechanism)

    print("Evaluation Metrics:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
