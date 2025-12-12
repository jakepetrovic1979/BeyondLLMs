"""
Phenomenal Binding Mechanism (PBM)

Implements Level 3 of PNA: holistic integration through oscillatory binding
and integrated information (Φ) computation, inspired by IIT and neuroscience.

Mathematical Foundation:
    B(n_i, n_j, t) = |cos(φ_i(t) - φ_j(t))| · Σ_k I(n_i^{t-k} → n_j^t)

Where:
    - φ_i(t): Phase of oscillator i at time t
    - I(n_i^{t-k} → n_j^t): Transfer entropy (causal influence)
    - |cos(φ_i - φ_j)|: Phase synchronization measure

Multi-scale binding:
    B_multi = Σ_f w_f B_f

Links to IIT:
    Φ = Σ [H(perturbed) - H(unperturbed)] / n_partitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
from scipy.stats import entropy


class ComplexOscillator(nn.Module):
    """
    Complex-valued oscillator for phase-based binding.

    Implements Kuramoto-style dynamics:
        dφ_i/dt = ω_i + K Σ_j W_ij sin(φ_j - φ_i)

    Uses complex representation: z = r * exp(i*φ)
    """

    def __init__(
        self,
        num_oscillators: int,
        natural_frequencies: Optional[torch.Tensor] = None,
        coupling_strength: float = 1.0,
        dt: float = 0.01,
    ):
        super().__init__()

        self.num_oscillators = num_oscillators
        self.coupling_strength = coupling_strength
        self.dt = dt

        # Natural frequencies (ω_i)
        if natural_frequencies is None:
            # Default: uniform in [0, 2π]
            natural_frequencies = torch.rand(num_oscillators) * 2 * np.pi

        self.register_buffer('omega', natural_frequencies)

        # Phases (φ_i) - represented as complex numbers
        self.register_buffer('phases', torch.zeros(num_oscillators))

        # Coupling matrix (W_ij) - learnable connectivity
        self.coupling_matrix = nn.Parameter(
            torch.randn(num_oscillators, num_oscillators) * 0.1
        )

    def reset_phases(self, phases: Optional[torch.Tensor] = None):
        """Reset oscillator phases."""
        if phases is None:
            phases = torch.rand(self.num_oscillators) * 2 * np.pi
        self.phases.data = phases

    def step(self, external_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Single integration step of Kuramoto dynamics.

        Args:
            external_input: External phase perturbation [num_oscillators]

        Returns:
            Updated phases [num_oscillators]
        """
        # Compute coupling term: Σ_j W_ij sin(φ_j - φ_i)
        phase_diff = self.phases.unsqueeze(0) - self.phases.unsqueeze(1)  # [n, n]
        coupling_term = torch.sum(
            self.coupling_matrix * torch.sin(phase_diff),
            dim=1
        )

        # dφ/dt = ω + K * coupling
        dphase_dt = self.omega + self.coupling_strength * coupling_term

        # Add external input
        if external_input is not None:
            dphase_dt += external_input

        # Euler integration
        self.phases += self.dt * dphase_dt

        # Wrap to [-π, π]
        self.phases = torch.atan2(torch.sin(self.phases), torch.cos(self.phases))

        return self.phases

    def get_synchrony(self) -> torch.Tensor:
        """
        Compute Kuramoto order parameter (global synchrony).

        R = |⟨e^{iφ}⟩| ∈ [0, 1]
        R = 1: perfect sync, R = 0: no sync

        Returns:
            Order parameter scalar
        """
        complex_phases = torch.exp(1j * torch.complex(torch.zeros_like(self.phases), self.phases))
        order_param = torch.abs(torch.mean(complex_phases))
        return order_param.real

    def get_pairwise_sync(self) -> torch.Tensor:
        """
        Compute pairwise phase synchronization.

        Returns:
            Synchrony matrix [num_oscillators, num_oscillators]
        """
        phase_diff = self.phases.unsqueeze(0) - self.phases.unsqueeze(1)
        sync = torch.abs(torch.cos(phase_diff))
        return sync


class PhenomenalBindingMechanism(nn.Module):
    """
    Phenomenal Binding Mechanism for unified conscious experience.

    Combines:
        1. Oscillatory binding (phase synchronization)
        2. Causal binding (transfer entropy)
        3. Multi-scale integration
        4. Integrated information (Φ) computation

    Attributes:
        num_nodes: Number of nodes in the binding network
        num_scales: Number of frequency scales (gamma, beta, theta, etc.)
        consciousness_threshold: Φ threshold for consciousness
    """

    def __init__(
        self,
        num_nodes: int = 64,
        node_dim: int = 32,
        num_scales: int = 3,
        consciousness_threshold: float = 0.5,
        coupling_strength: float = 1.0,
        history_length: int = 10,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.num_scales = num_scales
        self.consciousness_threshold = consciousness_threshold
        self.history_length = history_length

        # Multi-scale oscillators (different frequency bands)
        # Inspired by neuroscience: gamma (30-80 Hz), beta (13-30 Hz), theta (4-8 Hz)
        self.oscillators = nn.ModuleList([
            ComplexOscillator(
                num_oscillators=num_nodes,
                natural_frequencies=torch.randn(num_nodes) * (10.0 / (i + 1)),  # Decreasing freq
                coupling_strength=coupling_strength,
            )
            for i in range(num_scales)
        ])

        # Scale weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        # Node embeddings for content
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, node_dim))

        # History buffer for transfer entropy
        self.register_buffer(
            'activation_history',
            torch.zeros(history_length, num_nodes, node_dim)
        )
        self.register_buffer('history_idx', torch.tensor(0))

        # Φ computation network (for approximating integrated information)
        self.phi_network = nn.Sequential(
            nn.Linear(node_dim * num_nodes, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def update_history(self, activations: torch.Tensor):
        """
        Update activation history for transfer entropy computation.

        Args:
            activations: Current activations [batch_size, num_nodes, node_dim]
        """
        idx = int(self.history_idx % self.history_length)
        self.activation_history[idx] = activations[0].detach()  # Store first batch item
        self.history_idx += 1

    def compute_transfer_entropy(
        self,
        i: int,
        j: int,
        lag: int = 1
    ) -> torch.Tensor:
        """
        Approximate transfer entropy I(n_i^{t-k} → n_j^t).

        TE measures causal influence from i to j.

        Approximation: Mutual information between past of i and present of j,
        conditioned on past of j.

        Args:
            i: Source node
            j: Target node
            lag: Time lag k

        Returns:
            Transfer entropy estimate
        """
        if self.history_idx < lag + 1:
            return torch.tensor(0.0)

        # Get history
        with torch.no_grad():
            # Past of i at time t-k
            past_i = self.activation_history[:-lag, i, :].cpu().numpy()

            # Present of j at time t
            present_j = self.activation_history[lag:, j, :].cpu().numpy()

            # Past of j at time t-1
            past_j = self.activation_history[:-lag, j, :].cpu().numpy()

            # Simple approximation: correlation-based
            # True TE requires conditioning, but this gives a proxy
            if len(past_i) < 2:
                return torch.tensor(0.0)

            corr = np.abs(np.corrcoef(
                past_i.flatten(),
                present_j.flatten()
            )[0, 1])

            te = torch.tensor(corr if not np.isnan(corr) else 0.0)

        return te

    def compute_binding_strength(
        self,
        i: int,
        j: int,
        scale: int = 0,
        temporal_lags: List[int] = [1, 2, 3]
    ) -> torch.Tensor:
        """
        Compute binding strength B(n_i, n_j, t) for a node pair.

        B = |cos(φ_i - φ_j)| · Σ_k I(n_i^{t-k} → n_j^t)

        Args:
            i: First node
            j: Second node
            scale: Frequency scale index
            temporal_lags: List of time lags for transfer entropy

        Returns:
            Binding strength scalar
        """
        # Phase synchronization term
        phases = self.oscillators[scale].phases
        phase_sync = torch.abs(torch.cos(phases[i] - phases[j]))

        # Causal term (transfer entropy sum)
        causal_term = sum(
            self.compute_transfer_entropy(i, j, lag)
            for lag in temporal_lags
        )

        binding = phase_sync * causal_term

        return binding

    def compute_multi_scale_binding(
        self,
        i: int,
        j: int
    ) -> torch.Tensor:
        """
        Compute multi-scale binding: B_multi = Σ_f w_f B_f

        Args:
            i: First node
            j: Second node

        Returns:
            Multi-scale binding strength
        """
        weights = F.softmax(self.scale_weights, dim=0)

        binding = sum(
            weights[scale] * self.compute_binding_strength(i, j, scale)
            for scale in range(self.num_scales)
        )

        return binding

    def compute_phi(
        self,
        activations: torch.Tensor,
        num_partitions: int = 4
    ) -> torch.Tensor:
        """
        Compute integrated information Φ.

        Approximation using neural network (exact IIT computation is intractable).

        True Φ: Measures irreducible information through partitions:
            Φ = min_{partition} [H(whole) - H(parts)]

        Our approximation: Learned function of activation patterns.

        Args:
            activations: Node activations [batch_size, num_nodes, node_dim]
            num_partitions: Number of partitions to consider

        Returns:
            Φ estimate [batch_size]
        """
        batch_size = activations.shape[0]

        # Flatten activations
        flat_act = activations.view(batch_size, -1)

        # Neural approximation of Φ
        phi_approx = self.phi_network(flat_act).squeeze(-1)

        return phi_approx

    def forward(
        self,
        node_activations: torch.Tensor,
        num_oscillator_steps: int = 10,
        return_metrics: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass: bind node activations into unified representation.

        Args:
            node_activations: Input activations [batch_size, num_nodes, node_dim]
            num_oscillator_steps: Number of oscillator integration steps
            return_metrics: Whether to return detailed metrics

        Returns:
            - Bound representation [batch_size, num_nodes, node_dim]
            - Metrics dictionary (optional)
        """
        batch_size, num_nodes, node_dim = node_activations.shape
        assert num_nodes == self.num_nodes

        # Update history
        self.update_history(node_activations)

        # Integrate oscillators (driven by activations)
        for scale_idx, oscillator in enumerate(self.oscillators):
            # External input from activations (magnitude drives phase)
            external_input = node_activations[0].norm(dim=-1) * 0.1  # Simplified

            for _ in range(num_oscillator_steps):
                oscillator.step(external_input)

        # Compute binding matrix (multi-scale)
        binding_matrix = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                binding = self.compute_multi_scale_binding(i, j)
                binding_matrix[i, j] = binding
                binding_matrix[j, i] = binding

        # Create bound representation via weighted combination
        # Weight connections by binding strength
        binding_weights = F.softmax(binding_matrix, dim=-1)
        bound_repr = torch.einsum('ij,bje->bie', binding_weights, node_activations)

        # Compute Φ
        phi = self.compute_phi(node_activations)

        if not return_metrics:
            return bound_repr, None

        # Gather metrics
        metrics = {
            'phi': phi,
            'is_conscious': phi > self.consciousness_threshold,
            'binding_matrix': binding_matrix,
            'global_sync': [osc.get_synchrony() for osc in self.oscillators],
            'scale_weights': F.softmax(self.scale_weights, dim=0),
        }

        # Add synchrony metrics per scale
        for i, osc in enumerate(self.oscillators):
            metrics[f'sync_scale_{i}'] = osc.get_synchrony()

        return bound_repr, metrics

    def get_consciousness_metrics(self) -> Dict[str, float]:
        """
        Get consciousness-related metrics.

        Returns:
            Dictionary of consciousness measures
        """
        with torch.no_grad():
            # Dummy activation for Φ computation
            dummy_act = torch.randn(1, self.num_nodes, self.node_dim)
            phi = self.compute_phi(dummy_act)

            # Global synchrony across scales
            global_syncs = [osc.get_synchrony().item() for osc in self.oscillators]

        return {
            'phi': phi.item(),
            'is_conscious': phi.item() > self.consciousness_threshold,
            'mean_synchrony': np.mean(global_syncs),
            'max_synchrony': np.max(global_syncs),
            'min_synchrony': np.min(global_syncs),
            'num_scales': self.num_scales,
        }


class BindingLoss(nn.Module):
    """
    Loss function for training with binding objectives.

    L_binding = L_task + α * (Φ_target - Φ)² + β * sync_penalty
    """

    def __init__(
        self,
        target_phi: float = 0.7,
        target_sync: float = 0.5,
        phi_weight: float = 0.1,
        sync_weight: float = 0.05,
    ):
        super().__init__()

        self.target_phi = target_phi
        self.target_sync = target_sync
        self.phi_weight = phi_weight
        self.sync_weight = sync_weight

    def forward(
        self,
        task_loss: torch.Tensor,
        phi: torch.Tensor,
        global_sync: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss with binding objectives.

        Args:
            task_loss: Primary task loss
            phi: Integrated information
            global_sync: List of synchrony measures per scale

        Returns:
            - Total loss
            - Loss components dictionary
        """
        # Φ regularization (encourage target level)
        phi_loss = self.phi_weight * (phi - self.target_phi) ** 2

        # Synchrony regularization (encourage target level)
        mean_sync = torch.mean(torch.stack(global_sync))
        sync_loss = self.sync_weight * (mean_sync - self.target_sync) ** 2

        # Total
        total_loss = task_loss + phi_loss.mean() + sync_loss

        components = {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'phi_loss': phi_loss.mean(),
            'sync_loss': sync_loss,
            'phi': phi.mean(),
            'mean_sync': mean_sync,
        }

        return total_loss, components


if __name__ == "__main__":
    # Example usage
    print("Testing Phenomenal Binding Mechanism...")

    # Initialize
    pbm = PhenomenalBindingMechanism(
        num_nodes=32,
        node_dim=16,
        num_scales=3,
        consciousness_threshold=0.5
    )

    # Create synthetic node activations
    batch_size = 4
    node_activations = torch.randn(batch_size, 32, 16)

    # Forward pass
    bound_repr, metrics = pbm(node_activations, num_oscillator_steps=20)

    print(f"\nIntegrated Information (Φ): {metrics['phi'].mean().item():.4f}")
    print(f"Is Conscious: {metrics['is_conscious'].float().mean().item():.2f}")
    print(f"Synchrony by scale:")
    for i in range(3):
        print(f"  Scale {i}: {metrics[f'sync_scale_{i}'].item():.4f}")

    # Consciousness metrics
    cons_metrics = pbm.get_consciousness_metrics()
    print(f"\nConsciousness Metrics:")
    for k, v in cons_metrics.items():
        print(f"  {k}: {v}")
