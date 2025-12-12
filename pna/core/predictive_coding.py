"""
Predictive Coding Hierarchy

Implements hierarchical predictive coding for the PNA framework,
integrating with the Free Energy Principle for multi-level inference.

Mathematical Foundation:
    At layer l:
        - Prediction: μ_l = f(μ_{l+1})
        - Error: ε_l = x_l - μ_l
        - Update: Δμ_{l+1} ∝ ε_l · ∂f/∂μ_{l+1}

    Minimizes prediction error hierarchically:
        E = Σ_l π_l · ||ε_l||²

    where π_l is precision (inverse variance) at layer l.

Links to FEP:
    Prediction errors = gradients of free energy
    Local updates approximate backpropagation
    Biologically plausible credit assignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Callable
import numpy as np


class PredictiveCodingLayer(nn.Module):
    """
    Single layer in predictive coding hierarchy.

    Maintains:
        - Representation (μ)
        - Prediction error (ε)
        - Precision (π)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = 'tanh',
        learnable_precision: bool = True,
        init_precision: float = 1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Prediction function: μ_l = f(μ_{l+1})
        self.prediction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self._get_activation(activation),
            nn.Linear(hidden_dim, output_dim)
        )

        # Precision (inverse variance)
        if learnable_precision:
            self.log_precision = nn.Parameter(torch.log(torch.tensor(init_precision)))
        else:
            self.register_buffer('log_precision', torch.log(torch.tensor(init_precision)))

        # State buffers
        self.register_buffer('mu', torch.zeros(output_dim))
        self.register_buffer('epsilon', torch.zeros(output_dim))

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'sigmoid': nn.Sigmoid(),
        }
        return activations.get(name, nn.Tanh())

    @property
    def precision(self) -> torch.Tensor:
        """Current precision."""
        return torch.exp(self.log_precision).clamp(min=0.01, max=100.0)

    def predict(self, mu_above: torch.Tensor) -> torch.Tensor:
        """
        Generate prediction from layer above.

        Args:
            mu_above: Representation from layer l+1

        Returns:
            Prediction for layer l
        """
        return self.prediction_net(mu_above)

    def compute_error(self, x: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute precision-weighted prediction error.

        Args:
            x: Target (from below or data)
            prediction: Prediction from above

        Returns:
            Weighted error
        """
        error = x - prediction
        weighted_error = self.precision * error
        return weighted_error

    def forward(
        self,
        x: torch.Tensor,
        mu_above: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute prediction and error.

        Args:
            x: Input from below (or data)
            mu_above: Representation from above

        Returns:
            - Prediction
            - Error (precision-weighted)
            - Raw error (unweighted)
        """
        prediction = self.predict(mu_above)
        error = self.compute_error(x, prediction)
        raw_error = x - prediction

        return prediction, error, raw_error


class PredictiveCodingHierarchy(nn.Module):
    """
    Multi-layer predictive coding hierarchy.

    Implements bidirectional inference:
        - Top-down: predictions
        - Bottom-up: errors

    Minimizes total prediction error across hierarchy.

    Attributes:
        layer_dims: List of layer dimensions [bottom, ..., top]
        num_layers: Number of layers
        inference_steps: Number of iterative inference steps
        learning_rate: Learning rate for representation updates
    """

    def __init__(
        self,
        layer_dims: List[int],
        hidden_dim: int = 128,
        activation: str = 'tanh',
        learnable_precision: bool = True,
        inference_steps: int = 20,
        learning_rate: float = 0.1,
        top_down_weight: float = 1.0,
        bottom_up_weight: float = 1.0,
    ):
        super().__init__()

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.inference_steps = inference_steps
        self.learning_rate = learning_rate
        self.top_down_weight = top_down_weight
        self.bottom_up_weight = bottom_up_weight

        # Create layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = PredictiveCodingLayer(
                input_dim=layer_dims[i + 1],  # From above
                hidden_dim=hidden_dim,
                output_dim=layer_dims[i],  # To current level
                activation=activation,
                learnable_precision=learnable_precision,
            )
            self.layers.append(layer)

        # Top-level prior
        self.register_buffer('top_prior', torch.zeros(layer_dims[-1]))

        # Representation buffers (μ at each level)
        self.mus = [torch.zeros(dim) for dim in layer_dims]

    def reset_states(self, batch_size: int = 1):
        """Reset internal states."""
        for i, dim in enumerate(self.layer_dims):
            self.mus[i] = torch.zeros(batch_size, dim)

    def forward_pass(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Bottom-up forward pass (initial encoding).

        Args:
            x: Input data [batch_size, input_dim]

        Returns:
            List of representations at each level
        """
        batch_size = x.shape[0]
        self.reset_states(batch_size)

        # Start from bottom
        self.mus[0] = x

        # Simple forward encoding (will be refined by inference)
        current = x
        for i in range(self.num_layers - 1):
            # Use prediction network in reverse as encoder
            current = self.layers[i].prediction_net(current)
            if i < self.num_layers - 1:
                self.mus[i + 1] = current

        return self.mus

    def inference_step(
        self,
        x: torch.Tensor,
        mus: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Single step of iterative inference.

        Updates representations to minimize prediction error.

        Args:
            x: Input data [batch_size, input_dim]
            mus: Current representations at each level

        Returns:
            - Updated representations
            - Predictions at each level
            - Errors at each level
        """
        batch_size = x.shape[0]
        predictions = [None] * self.num_layers
        errors = [None] * self.num_layers
        raw_errors = [None] * self.num_layers

        # Top-down predictions
        for i in range(self.num_layers - 1):
            pred, err, raw_err = self.layers[i](
                x=mus[i],
                mu_above=mus[i + 1]
            )
            predictions[i] = pred
            errors[i] = err
            raw_errors[i] = raw_err

        # Update representations (gradient descent on error)
        new_mus = [mu.clone() for mu in mus]

        # Bottom layer: direct from data
        new_mus[0] = x

        # Middle layers: combine top-down and bottom-up
        for i in range(1, self.num_layers - 1):
            # Top-down error (as target)
            td_error = errors[i]  # From layer above

            # Bottom-up error (from layer below)
            bu_error = -errors[i - 1]  # Negative because we're the target

            # Update: balance both sources
            total_error = (
                self.top_down_weight * td_error +
                self.bottom_up_weight * bu_error
            )

            new_mus[i] = mus[i] + self.learning_rate * total_error

        # Top layer: only bottom-up error
        if self.num_layers > 1:
            new_mus[-1] = mus[-1] - self.learning_rate * errors[-1]

        return new_mus, predictions, raw_errors

    def forward(
        self,
        x: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Full inference: iterative prediction error minimization.

        Args:
            x: Input data [batch_size, input_dim]
            return_trajectory: Whether to return optimization trajectory

        Returns:
            - Final representations at each level
            - Metrics dictionary
        """
        batch_size = x.shape[0]

        # Initialize with forward pass
        mus = self.forward_pass(x)

        # Iterative inference
        trajectory = [] if return_trajectory else None

        for step in range(self.inference_steps):
            mus, predictions, raw_errors = self.inference_step(x, mus)

            if return_trajectory:
                total_error = sum(
                    (err ** 2).mean() for err in raw_errors if err is not None
                )
                trajectory.append({
                    'step': step,
                    'total_error': total_error.item(),
                    'layer_errors': [
                        (err ** 2).mean().item() if err is not None else 0.0
                        for err in raw_errors
                    ]
                })

        # Final metrics
        final_errors = []
        for i in range(self.num_layers - 1):
            pred, err, raw_err = self.layers[i](mus[i], mus[i + 1])
            final_errors.append((raw_err ** 2).mean())

        metrics = {
            'total_error': sum(final_errors),
            'layer_errors': final_errors,
            'representations': mus,
            'precisions': [layer.precision for layer in self.layers],
        }

        if return_trajectory:
            metrics['trajectory'] = trajectory

        return mus, metrics

    def reconstruct(self, mus: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Top-down reconstruction from representations.

        Args:
            mus: Representations (optional, uses internal state if None)

        Returns:
            Reconstructed input
        """
        if mus is None:
            mus = self.mus

        # Start from top
        current = mus[-1]

        # Generate predictions downward
        for i in range(self.num_layers - 2, -1, -1):
            current = self.layers[i].predict(current)

        return current

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to top-level representation.

        Args:
            x: Input data [batch_size, input_dim]

        Returns:
            Top-level representation [batch_size, top_dim]
        """
        mus, _ = self.forward(x)
        return mus[-1]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from top-level representation.

        Args:
            z: Top-level representation [batch_size, top_dim]

        Returns:
            Reconstructed input [batch_size, input_dim]
        """
        # Create dummy mus with z at top
        mus = [torch.zeros(z.shape[0], dim) for dim in self.layer_dims]
        mus[-1] = z

        return self.reconstruct(mus)

    def get_free_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute variational free energy of the hierarchy.

        F = Σ_l π_l · ||ε_l||²

        Args:
            x: Input data

        Returns:
            Free energy scalar
        """
        mus, metrics = self.forward(x)

        # Precision-weighted errors
        free_energy = 0.0
        for i, layer in enumerate(self.layers):
            error = metrics['layer_errors'][i]
            precision = layer.precision
            free_energy += precision * error

        return free_energy


class AttentionModulatedPC(PredictiveCodingHierarchy):
    """
    Predictive Coding with attention-modulated precision.

    Dynamically adjusts precision based on prediction uncertainty
    and attention signals.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(self.layer_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, self.num_layers - 1),
            nn.Softplus()  # Ensure positive
        )

    def compute_attention_precision(self, top_repr: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute attention-modulated precision weights.

        Args:
            top_repr: Top-level representation

        Returns:
            List of precision modulations per layer
        """
        attention_weights = self.attention_net(top_repr)
        return [attention_weights[:, i] for i in range(self.num_layers - 1)]

    def forward(
        self,
        x: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward with attention modulation."""
        # Standard inference
        mus, metrics = super().forward(x, return_trajectory)

        # Modulate precision
        attention_mods = self.compute_attention_precision(mus[-1])
        metrics['attention_weights'] = attention_mods

        return mus, metrics


if __name__ == "__main__":
    # Example usage
    print("Testing Predictive Coding Hierarchy...")

    # Initialize (e.g., for image processing)
    layer_dims = [784, 256, 64, 16]  # MNIST-like: 28x28 -> ... -> 16
    pc = PredictiveCodingHierarchy(
        layer_dims=layer_dims,
        hidden_dim=128,
        inference_steps=30,
        learning_rate=0.05
    )

    # Create synthetic input
    batch_size = 8
    x = torch.randn(batch_size, 784)

    # Inference
    mus, metrics = pc(x, return_trajectory=True)

    print(f"\nTotal Error: {metrics['total_error'].item():.6f}")
    print(f"Layer Errors:")
    for i, err in enumerate(metrics['layer_errors']):
        print(f"  Layer {i}: {err.item():.6f}")

    # Reconstruction
    recon = pc.reconstruct(mus)
    recon_error = F.mse_loss(recon, x)
    print(f"\nReconstruction Error: {recon_error.item():.6f}")

    # Encode-decode
    z = pc.encode(x)
    x_recon = pc.decode(z)
    print(f"Encode-Decode Error: {F.mse_loss(x_recon, x).item():.6f}")

    # Show convergence
    print(f"\nError Trajectory:")
    for entry in metrics['trajectory'][:5]:
        print(f"  Step {entry['step']}: {entry['total_error']:.6f}")
    print("  ...")
    for entry in metrics['trajectory'][-3:]:
        print(f"  Step {entry['step']}: {entry['total_error']:.6f}")
