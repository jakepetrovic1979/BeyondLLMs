"""
HOPE-like Nested Optimizer

Simplified implementation of Hierarchical Optimization for Persistent Evolution,
inspired by Behrouz et al. (2025) nested learning paradigm.

Implements meta-learning and self-referential optimization at multiple timescales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import copy


class HOPEOptimizer:
    """
    Hierarchical Optimization for Persistent Evolution.

    Multi-timescale optimizer that learns to optimize itself.

    Timescales:
        - Fast: Parameter updates (standard gradient descent)
        - Medium: Learning rate adaptation (meta-learning)
        - Slow: Architecture modification (structural learning)

    Attributes:
        model: Neural network to optimize
        fast_lr: Fast timescale learning rate
        meta_lr: Meta-learning rate
        structural_lr: Structural modification rate
    """

    def __init__(
        self,
        model: nn.Module,
        fast_lr: float = 1e-3,
        meta_lr: float = 1e-4,
        structural_lr: float = 1e-5,
        enable_meta: bool = True,
        enable_structural: bool = False,
    ):
        self.model = model
        self.fast_lr = fast_lr
        self.meta_lr = meta_lr
        self.structural_lr = structural_lr
        self.enable_meta = enable_meta
        self.enable_structural = enable_structural

        # Fast optimizer (standard)
        self.fast_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=fast_lr
        )

        # Meta-parameters (learnable learning rates)
        if enable_meta:
            self.meta_params = {
                name: nn.Parameter(torch.tensor(fast_lr))
                for name, _ in model.named_parameters()
            }
            self.meta_optimizer = torch.optim.Adam(
                self.meta_params.values(),
                lr=meta_lr
            )

        # History for continual learning
        self.loss_history = []
        self.gradient_history = []

    def fast_step(self, loss: torch.Tensor):
        """
        Fast timescale update (standard gradient descent).

        Args:
            loss: Current loss
        """
        self.fast_optimizer.zero_grad()
        loss.backward(create_graph=self.enable_meta)  # Create graph for meta-learning

        # Store gradients
        if self.enable_meta:
            grads = {
                name: param.grad.clone() if param.grad is not None else None
                for name, param in self.model.named_parameters()
            }
            self.gradient_history.append(grads)

        self.fast_optimizer.step()

        # Track loss
        self.loss_history.append(loss.item())

    def meta_step(self, validation_loss: torch.Tensor):
        """
        Meta-learning step: optimize the optimizer.

        Updates learning rates based on validation performance.

        Args:
            validation_loss: Validation loss for meta-learning
        """
        if not self.enable_meta:
            return

        self.meta_optimizer.zero_grad()

        # Meta-objective: minimize validation loss w.r.t. meta-parameters
        validation_loss.backward()

        self.meta_optimizer.step()

        # Apply updated learning rates
        for (name, param), (meta_name, meta_param) in zip(
            self.model.named_parameters(),
            self.meta_params.items()
        ):
            # Update param group learning rate
            for group in self.fast_optimizer.param_groups:
                for p in group['params']:
                    if p is param:
                        group['lr'] = torch.abs(meta_param).item()

    def structural_step(self):
        """
        Structural learning: modify architecture.

        Placeholder for architecture search / neural architecture meta-learning.
        """
        if not self.enable_structural:
            return

        # Placeholder: could implement:
        # - Pruning low-magnitude weights
        # - Growing new connections
        # - Modifying layer sizes
        pass

    def get_learning_rates(self) -> Dict[str, float]:
        """Get current learning rates for all parameters."""
        if self.enable_meta:
            return {
                name: torch.abs(meta_param).item()
                for name, meta_param in self.meta_params.items()
            }
        else:
            return {'global': self.fast_lr}

    def get_statistics(self) -> Dict[str, float]:
        """Get optimizer statistics."""
        stats = {
            'mean_loss': torch.tensor(self.loss_history[-100:]).mean().item() if self.loss_history else 0.0,
            'loss_std': torch.tensor(self.loss_history[-100:]).std().item() if len(self.loss_history) > 1 else 0.0,
        }

        if self.enable_meta:
            lrs = list(self.get_learning_rates().values())
            stats['mean_lr'] = sum(lrs) / len(lrs) if lrs else self.fast_lr
            stats['lr_std'] = torch.tensor(lrs).std().item() if len(lrs) > 1 else 0.0

        return stats


class NestedLearningModule(nn.Module):
    """
    Neural module with built-in nested learning capabilities.

    Combines fast learning (gradient descent) with slow learning
    (meta-parameters, architectural changes).
    """

    def __init__(
        self,
        base_module: nn.Module,
        enable_meta: bool = True,
        enable_continual: bool = True,
    ):
        super().__init__()

        self.base_module = base_module
        self.enable_meta = enable_meta
        self.enable_continual = enable_continual

        # Meta-learning: learnable transformation on gradients
        if enable_meta:
            # Simple version: scale and shift gradients
            self.gradient_transform = nn.Sequential(
                nn.Linear(1, 32),
                nn.Tanh(),
                nn.Linear(32, 2)  # [scale, shift]
            )

        # Continual learning: elastic weight consolidation
        if enable_continual:
            self.register_buffer(
                'fisher_information',
                torch.zeros_like(next(base_module.parameters()))
            )
            self.register_buffer(
                'optimal_params',
                next(base_module.parameters()).clone()
            )

    def forward(self, *args, **kwargs):
        """Forward through base module."""
        return self.base_module(*args, **kwargs)

    def meta_transform_gradients(self):
        """Apply meta-learned transformation to gradients."""
        if not self.enable_meta:
            return

        for param in self.base_module.parameters():
            if param.grad is None:
                continue

            # Get gradient magnitude
            grad_mag = param.grad.norm().unsqueeze(0)

            # Meta-learned transform
            scale_shift = self.gradient_transform(grad_mag)
            scale = torch.sigmoid(scale_shift[0])  # [0, 1]
            shift = scale_shift[1] * 0.1

            # Apply
            param.grad = param.grad * scale + shift

    def consolidate_knowledge(self, data_loader, loss_fn, num_samples: int = 100):
        """
        Consolidate knowledge for continual learning (EWC).

        Computes Fisher information matrix to protect important weights.

        Args:
            data_loader: Data loader for computing Fisher
            loss_fn: Loss function
            num_samples: Number of samples for estimation
        """
        if not self.enable_continual:
            return

        fisher = torch.zeros_like(next(self.base_module.parameters()))

        self.base_module.train()
        for i, (x, y) in enumerate(data_loader):
            if i >= num_samples:
                break

            # Compute loss
            output = self.base_module(x)
            loss = loss_fn(output, y)

            # Compute gradients
            self.base_module.zero_grad()
            loss.backward()

            # Accumulate squared gradients (diagonal Fisher approximation)
            for param in self.base_module.parameters():
                if param.grad is not None:
                    fisher += param.grad ** 2

        # Average
        fisher /= num_samples

        # Store
        self.fisher_information = fisher
        self.optimal_params = next(self.base_module.parameters()).clone()

    def ewc_loss(self, lambda_ewc: float = 1000.0) -> torch.Tensor:
        """
        Compute Elastic Weight Consolidation loss.

        Penalizes changes to important weights.

        Args:
            lambda_ewc: Regularization strength

        Returns:
            EWC penalty
        """
        if not self.enable_continual:
            return torch.tensor(0.0)

        loss = 0.0
        for param, optimal_param in zip(
            self.base_module.parameters(),
            [self.optimal_params]
        ):
            loss += (self.fisher_information * (param - optimal_param) ** 2).sum()

        return lambda_ewc * loss


if __name__ == "__main__":
    # Example usage
    print("Testing HOPE Optimizer...")

    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )

    # Initialize HOPE
    optimizer = HOPEOptimizer(
        model=model,
        fast_lr=1e-3,
        meta_lr=1e-4,
        enable_meta=True,
        enable_structural=False,
    )

    # Synthetic training
    for step in range(10):
        # Create batch
        x = torch.randn(16, 10)
        y = torch.randint(0, 2, (16,))

        # Forward
        output = model(x)
        loss = F.cross_entropy(output, y)

        # Fast step
        optimizer.fast_step(loss)

        print(f"Step {step}: Loss = {loss.item():.4f}")

        # Meta step (every N steps)
        if step % 5 == 0 and step > 0:
            # Validation batch
            x_val = torch.randn(16, 10)
            y_val = torch.randint(0, 2, (16,))
            output_val = model(x_val)
            val_loss = F.cross_entropy(output_val, y_val)

            optimizer.meta_step(val_loss)
            print(f"  Meta-step: Val Loss = {val_loss.item():.4f}")

    # Statistics
    stats = optimizer.get_statistics()
    print(f"\nOptimizer Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")

    # Learning rates
    lrs = optimizer.get_learning_rates()
    print(f"\nLearning Rates (first 5):")
    for i, (k, v) in enumerate(list(lrs.items())[:5]):
        print(f"  {k}: {v:.6f}")
