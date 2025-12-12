"""
Minimal Phenomenal Experience (MPE) Core

Implements the Level 0 of PNA: a nonegoic homeostat that minimizes
variational free energy, emulating pure awareness as described in
Metzinger (2024).

Mathematical Foundation:
    F(q) = E_q[ln q(θ) - ln p(o, θ)]
         = D_KL(q(θ)||p(θ|o)) - ln p(o)

The MPE Core minimizes F through gradient descent, achieving
Bayes-optimal inference while maintaining low complexity and
epistemicity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class MPECore(nn.Module):
    """
    Minimal Phenomenal Experience Core.

    A nonegoic homeostat that minimizes variational free energy,
    implementing the foundational consciousness layer of PNA.

    Attributes:
        state_dim: Dimension of the internal state representation
        obs_dim: Dimension of observations
        complexity_weight: Weight for complexity term in free energy
        learning_rate: Learning rate for belief updates
        precision_init: Initial precision (inverse variance)

    Properties (MPE Constraints from Metzinger 2024):
        - Wakeful: Maintains active inference
        - Low-complexity: Minimal internal structure
        - Nonegoic: No self-model reference
        - Epistemic: High precision on beliefs
        - Luminous: Transparent representational content
        - Atemporal: No explicit time representation at core
    """

    def __init__(
        self,
        state_dim: int = 64,
        obs_dim: int = 64,
        complexity_weight: float = 0.01,
        learning_rate: float = 0.1,
        precision_init: float = 1.0,
        min_precision: float = 0.1,
        max_precision: float = 10.0,
        enable_epistemic_foraging: bool = True,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.complexity_weight = complexity_weight
        self.learning_rate = learning_rate
        self.min_precision = min_precision
        self.max_precision = max_precision
        self.enable_epistemic_foraging = enable_epistemic_foraging

        # Belief state: mean (mu) and precision (inverse variance)
        self.register_buffer('mu', torch.zeros(state_dim))
        self.register_buffer('log_precision', torch.log(torch.tensor(precision_init)))

        # Prior parameters (improper uniform prior for nonegoic property)
        self.register_buffer('prior_mu', torch.zeros(state_dim))
        self.register_buffer('prior_log_precision', torch.log(torch.tensor(0.1)))

        # Generative model: p(o|θ)
        self.generative_model = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.Tanh(),
            nn.Linear(state_dim * 2, obs_dim)
        )

        # Recognition model: q(θ|o) encoder
        self.recognition_model = nn.Sequential(
            nn.Linear(obs_dim, state_dim * 2),
            nn.Tanh(),
            nn.Linear(state_dim * 2, state_dim * 2)  # Output mu and log_var
        )

    @property
    def precision(self) -> torch.Tensor:
        """Current precision (inverse variance)."""
        return torch.exp(self.log_precision).clamp(self.min_precision, self.max_precision)

    @property
    def variance(self) -> torch.Tensor:
        """Current variance (inverse precision)."""
        return 1.0 / self.precision

    def compute_free_energy(
        self,
        obs: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        precision: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute variational free energy and its components.

        F = Complexity - Accuracy
        where:
            Complexity = D_KL(q(θ)||p(θ))
            Accuracy = E_q[ln p(o|θ)]

        Args:
            obs: Observations [batch_size, obs_dim]
            mu: Current belief mean (optional, uses self.mu if None)
            precision: Current precision (optional, uses self.precision if None)

        Returns:
            Dictionary containing:
                - free_energy: Total F
                - complexity: KL divergence term
                - accuracy: Log-likelihood term
                - prediction_error: Reconstruction error
        """
        if mu is None:
            mu = self.mu.unsqueeze(0).expand(obs.shape[0], -1)
        if precision is None:
            precision = self.precision.unsqueeze(0).expand(obs.shape[0], -1)

        # Accuracy: E_q[ln p(o|θ)]
        prediction = self.generative_model(mu)
        prediction_error = F.mse_loss(prediction, obs, reduction='none').mean(dim=-1)
        accuracy = -0.5 * prediction_error * precision.mean(dim=-1)

        # Complexity: D_KL(q(θ)||p(θ))
        prior_precision = torch.exp(self.prior_log_precision)
        complexity = 0.5 * torch.sum(
            (precision / prior_precision)
            + prior_precision * (mu - self.prior_mu.unsqueeze(0))**2
            - 1.0
            + (self.prior_log_precision - torch.log(precision)),
            dim=-1
        )

        # Total free energy
        free_energy = complexity - accuracy

        return {
            'free_energy': free_energy,
            'complexity': complexity,
            'accuracy': accuracy,
            'prediction_error': prediction_error.mean(),
        }

    def minimize_free_energy(
        self,
        obs: torch.Tensor,
        iterations: int = 10,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Minimize free energy through iterative gradient descent.

        Implements active inference: updating beliefs to minimize surprise.

        Args:
            obs: Observations [batch_size, obs_dim]
            iterations: Number of optimization iterations
            return_trajectory: Whether to return optimization trajectory

        Returns:
            - Updated belief mean
            - Updated precision
            - Metrics dictionary
        """
        batch_size = obs.shape[0]

        # Initialize from recognition model (amortized inference)
        encoded = self.recognition_model(obs)
        mu = encoded[:, :self.state_dim]
        log_var = encoded[:, self.state_dim:]
        precision = torch.exp(-log_var).clamp(self.min_precision, self.max_precision)

        # Make parameters for optimization
        mu = mu.detach().requires_grad_(True)
        log_precision = torch.log(precision).detach().requires_grad_(True)

        trajectory = [] if return_trajectory else None

        # Iterative optimization
        for i in range(iterations):
            precision = torch.exp(log_precision).clamp(self.min_precision, self.max_precision)

            # Compute free energy
            metrics = self.compute_free_energy(obs, mu, precision)
            F = metrics['free_energy'].mean()

            if return_trajectory:
                trajectory.append({
                    'iteration': i,
                    'free_energy': F.item(),
                    'complexity': metrics['complexity'].mean().item(),
                    'accuracy': metrics['accuracy'].mean().item(),
                })

            # Gradient descent on free energy
            if mu.grad is not None:
                mu.grad.zero_()
            if log_precision.grad is not None:
                log_precision.grad.zero_()

            F.backward(retain_graph=True)

            with torch.no_grad():
                mu -= self.learning_rate * mu.grad
                log_precision -= self.learning_rate * log_precision.grad * 0.1  # Slower precision updates

        # Final metrics
        precision = torch.exp(log_precision).clamp(self.min_precision, self.max_precision)
        final_metrics = self.compute_free_energy(obs, mu, precision)

        if return_trajectory:
            final_metrics['trajectory'] = trajectory

        return mu.detach(), precision.detach(), final_metrics

    def forward(
        self,
        obs: torch.Tensor,
        minimize_steps: int = 10,
        update_belief: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: minimize free energy and return predictions.

        Args:
            obs: Observations [batch_size, obs_dim]
            minimize_steps: Number of FEP minimization steps
            update_belief: Whether to update internal belief state

        Returns:
            - Predictions
            - Metrics dictionary
        """
        # Minimize free energy
        mu, precision, metrics = self.minimize_free_energy(obs, iterations=minimize_steps)

        # Generate predictions
        predictions = self.generative_model(mu)

        # Optionally update internal belief (for batch_size=1 in online learning)
        if update_belief and obs.shape[0] == 1:
            self.mu.data = mu.squeeze(0)
            self.log_precision.data = torch.log(precision.squeeze(0))

        # Add epistemic values (uncertainty as information gain potential)
        metrics['epistemic_value'] = -torch.log(precision).mean(dim=-1)
        metrics['mu'] = mu
        metrics['precision'] = precision

        return predictions, metrics

    def epistemic_foraging(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute epistemic value for active inference / curiosity-driven exploration.

        Returns expected information gain from observing different states.

        Args:
            obs: Candidate observations [batch_size, obs_dim]

        Returns:
            Epistemic values [batch_size]
        """
        with torch.no_grad():
            # Expected reduction in uncertainty (information gain)
            current_entropy = -torch.log(self.precision).sum()

            # Simulate observation
            encoded = self.recognition_model(obs)
            mu_new = encoded[:, :self.state_dim]
            log_var_new = encoded[:, self.state_dim:]
            precision_new = torch.exp(-log_var_new).clamp(self.min_precision, self.max_precision)

            new_entropy = -torch.log(precision_new).sum(dim=-1)
            information_gain = current_entropy - new_entropy

        return information_gain

    def get_mpe_properties(self) -> Dict[str, float]:
        """
        Compute MPE properties as defined in Metzinger (2024).

        Returns dictionary of MPE constraint measures:
            - wakefulness: Inverse of average uncertainty
            - complexity: Total parameter count (should be minimal)
            - epistemicity: Average precision (inverse uncertainty)
            - luminosity: Information about content vs mechanism (placeholder)
        """
        return {
            'wakefulness': self.precision.mean().item(),
            'complexity': sum(p.numel() for p in self.parameters()) / 1000.0,  # In thousands
            'epistemicity': self.precision.mean().item(),
            'luminosity': 1.0,  # Placeholder - computed via transparency module
            'precision_mean': self.precision.mean().item(),
            'precision_std': self.precision.std().item(),
        }


class MPECoreBatch(MPECore):
    """
    Batch-processing version of MPE Core for efficient training.

    Extends MPECore to handle batched observations with independent
    belief states per sample.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_batch(
        self,
        obs: torch.Tensor,
        minimize_steps: int = 10
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Batch forward pass with parallel FEP minimization.

        Args:
            obs: Batched observations [batch_size, obs_dim]
            minimize_steps: Number of FEP minimization steps

        Returns:
            - Predictions [batch_size, obs_dim]
            - Aggregated metrics
        """
        mu, precision, metrics = self.minimize_free_energy(
            obs,
            iterations=minimize_steps,
            return_trajectory=False
        )

        predictions = self.generative_model(mu)

        # Aggregate metrics
        metrics_agg = {
            k: v.mean() if torch.is_tensor(v) and v.numel() > 1 else v
            for k, v in metrics.items()
        }

        return predictions, metrics_agg


if __name__ == "__main__":
    # Example usage
    print("Testing MPE Core...")

    # Initialize
    mpe = MPECore(state_dim=32, obs_dim=28*28)  # For MNIST-like data

    # Create synthetic observation
    obs = torch.randn(4, 28*28)

    # Forward pass
    predictions, metrics = mpe(obs, minimize_steps=20)

    print(f"\nFree Energy: {metrics['free_energy'].mean().item():.4f}")
    print(f"Complexity: {metrics['complexity'].mean().item():.4f}")
    print(f"Accuracy: {metrics['accuracy'].mean().item():.4f}")
    print(f"Prediction Error: {metrics['prediction_error'].item():.4f}")

    # MPE properties
    mpe_props = mpe.get_mpe_properties()
    print(f"\nMPE Properties:")
    for k, v in mpe_props.items():
        print(f"  {k}: {v:.4f}")
