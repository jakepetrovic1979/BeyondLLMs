"""
Consciousness and PNA Metrics

Utilities for computing and tracking consciousness-related metrics:
- Integrated Information (Φ)
- Phase Synchronization
- Free Energy components
- Transparency measures
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ConsciousnessMetrics:
    """Container for consciousness metrics."""
    phi: float
    synchrony: float
    free_energy: float
    opacity: float
    is_conscious: bool
    timestamp: Optional[float] = None


def compute_phi_approximation(
    activations: torch.Tensor,
    method: str = 'variance'
) -> torch.Tensor:
    """
    Approximate integrated information Φ.

    Exact IIT computation is intractable, so we use approximations:
        - 'variance': Φ ≈ variance of activations (complexity)
        - 'correlation': Φ ≈ mean pairwise correlation (integration)
        - 'entropy': Φ ≈ entropy of activation distribution

    Args:
        activations: Node activations [batch_size, num_nodes, node_dim]
        method: Approximation method

    Returns:
        Φ estimate [batch_size]
    """
    batch_size, num_nodes, node_dim = activations.shape

    if method == 'variance':
        # Variance across nodes (differentiation)
        phi = activations.var(dim=1).mean(dim=-1)

    elif method == 'correlation':
        # Mean absolute pairwise correlation (integration)
        flat = activations.view(batch_size, num_nodes * node_dim)
        corr_matrix = torch.corrcoef(flat.T)
        phi = torch.abs(corr_matrix).mean(dim=[-2, -1])

    elif method == 'entropy':
        # Entropy approximation
        # Discretize activations and compute entropy
        bins = 10
        hist = torch.histc(activations.flatten(1), bins=bins, min=-3, max=3)
        probs = hist / hist.sum()
        probs = probs[probs > 0]  # Remove zeros
        entropy = -(probs * torch.log(probs)).sum()
        phi = entropy.expand(batch_size)

    else:
        raise ValueError(f"Unknown method: {method}")

    return phi


def compute_phase_synchrony(
    phases: torch.Tensor,
    mode: str = 'global'
) -> torch.Tensor:
    """
    Compute phase synchronization.

    Args:
        phases: Oscillator phases [num_oscillators]
        mode: 'global' or 'pairwise'

    Returns:
        Synchrony measure
    """
    if mode == 'global':
        # Kuramoto order parameter
        complex_phases = torch.exp(1j * torch.complex(torch.zeros_like(phases), phases))
        R = torch.abs(torch.mean(complex_phases))
        return R.real

    elif mode == 'pairwise':
        # Pairwise phase locking
        num_osc = phases.shape[0]
        sync_matrix = torch.zeros(num_osc, num_osc)

        for i in range(num_osc):
            for j in range(i+1, num_osc):
                phase_diff = phases[i] - phases[j]
                plv = torch.abs(torch.cos(phase_diff))  # Phase locking value
                sync_matrix[i, j] = plv
                sync_matrix[j, i] = plv

        return sync_matrix

    else:
        raise ValueError(f"Unknown mode: {mode}")


def track_consciousness_trajectory(
    model,
    data_loader,
    num_batches: int = 100
) -> List[ConsciousnessMetrics]:
    """
    Track consciousness metrics over time.

    Args:
        model: PNA model
        data_loader: Data loader
        num_batches: Number of batches to track

    Returns:
        List of ConsciousnessMetrics
    """
    trajectory = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break

            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            # Forward pass
            _, metrics = model(x, return_all_metrics=True)

            # Extract consciousness metrics
            cons_metric = ConsciousnessMetrics(
                phi=metrics['phi'].mean().item(),
                synchrony=metrics['mean_sync'].item(),
                free_energy=metrics['free_energy'].item(),
                opacity=metrics.get('opacity', 0.0).item() if 'opacity' in metrics else 0.0,
                is_conscious=metrics['is_conscious'].float().mean().item() > 0.5,
                timestamp=i,
            )

            trajectory.append(cons_metric)

    return trajectory


def compute_binding_strength(
    node1_act: torch.Tensor,
    node2_act: torch.Tensor,
    phase1: torch.Tensor,
    phase2: torch.Tensor
) -> float:
    """
    Compute binding strength between two nodes.

    B = phase_sync * causal_influence

    Args:
        node1_act: Activations of node 1 over time [T, dim]
        node2_act: Activations of node 2 over time [T, dim]
        phase1: Phase of oscillator 1
        phase2: Phase of oscillator 2

    Returns:
        Binding strength scalar
    """
    # Phase synchronization
    phase_sync = torch.abs(torch.cos(phase1 - phase2)).item()

    # Causal influence (cross-correlation)
    node1_flat = node1_act.flatten()
    node2_flat = node2_act.flatten()

    # Normalize
    node1_norm = (node1_flat - node1_flat.mean()) / (node1_flat.std() + 1e-8)
    node2_norm = (node2_flat - node2_flat.mean()) / (node2_flat.std() + 1e-8)

    # Cross-correlation
    causal = torch.dot(node1_norm, node2_norm).item() / len(node1_flat)

    # Binding
    binding = phase_sync * abs(causal)

    return binding


class MetricsLogger:
    """Logger for PNA training metrics."""

    def __init__(self):
        self.history = {
            'loss': [],
            'phi': [],
            'free_energy': [],
            'polarity_penalty': [],
            'synchrony': [],
            'is_conscious': [],
            'is_balanced': [],
        }

    def log(self, metrics: Dict):
        """Log a set of metrics."""
        for key in self.history.keys():
            if key in metrics:
                value = metrics[key]
                if torch.is_tensor(value):
                    value = value.item()
                self.history[key].append(value)

    def get_summary(self, window: int = 100) -> Dict:
        """Get summary statistics over recent window."""
        summary = {}

        for key, values in self.history.items():
            recent = values[-window:] if len(values) >= window else values

            if recent:
                summary[f'{key}_mean'] = np.mean(recent)
                summary[f'{key}_std'] = np.std(recent)
                summary[f'{key}_min'] = np.min(recent)
                summary[f'{key}_max'] = np.max(recent)

        return summary

    def save(self, filepath: str):
        """Save metrics to file."""
        np.savez(filepath, **{k: np.array(v) for k, v in self.history.items()})

    def load(self, filepath: str):
        """Load metrics from file."""
        data = np.load(filepath)
        self.history = {k: v.tolist() for k, v in data.items()}


@dataclass
class ContinualLearningMetrics:
    """Container for continual learning metrics."""
    A_final: float  # Final average accuracy
    F_avg: float  # Average forgetting
    BWT: float  # Backward transfer
    FWT: float  # Forward transfer
    n_fwt_measured: int  # Number of measured FWT cells
    accuracy_matrix: Optional[np.ndarray] = None


def compute_continual_learning_metrics(
    accuracy_matrix: np.ndarray,
    chance_level: float = 50.0,
    verbose: bool = False
) -> ContinualLearningMetrics:
    """
    Compute continual learning metrics with proper NaN handling.

    This implementation addresses reviewer concerns about:
    1. Diagonal-based vs max-over-time forgetting (we use diagonal)
    2. FWT computation from upper triangle (properly handles NaN)
    3. BWT = -F_avg relationship (explicit)

    Metrics:
        A_final = mean(A[T-1, :])  - final average accuracy
        F_avg = mean(A[j,j] - A[T-1,j]) for j < T  - average forgetting
        BWT = mean(A[T-1,j] - A[j,j]) for j < T  - backward transfer
        FWT = mean(A[j-1,j] - chance) for j >= 1  - forward transfer

    Args:
        accuracy_matrix: [T, T] accuracy matrix where A[i,j] is accuracy on
                        task j after training through task i. May contain NaN
                        for unmeasured cells (upper triangle).
        chance_level: Random chance baseline (default 50% for binary tasks)
        verbose: Print detailed breakdown

    Returns:
        ContinualLearningMetrics object

    Note:
        - Diagonal-based forgetting is valid for disjoint task sequences
          where peak performance occurs at diagonal
        - FWT only uses measured cells (non-NaN)
        - BWT = -F_avg by construction for diagonal-based forgetting
    """
    T = accuracy_matrix.shape[0]

    # Final accuracy: mean of last row (ignore NaN)
    final_row = accuracy_matrix[T-1, :]
    A_final = np.nanmean(final_row)

    # Forgetting: F_j = A[j,j] - A[T-1,j] for j < T
    forgetting = []
    if verbose:
        print("\nForgetting breakdown:")
    for j in range(T-1):
        diag = accuracy_matrix[j, j]
        final = accuracy_matrix[T-1, j]
        if not np.isnan(diag) and not np.isnan(final):
            f_j = diag - final
            forgetting.append(f_j)
            if verbose:
                print(f"  F_{j+1} = {diag:.2f} - {final:.2f} = {f_j:.2f}")
    F_avg = np.mean(forgetting) if len(forgetting) > 0 else np.nan

    # Backward Transfer: BWT = mean(A[T-1,j] - A[j,j]) for j < T
    bwt_terms = []
    for j in range(T-1):
        final = accuracy_matrix[T-1, j]
        diag = accuracy_matrix[j, j]
        if not np.isnan(final) and not np.isnan(diag):
            bwt_j = final - diag
            bwt_terms.append(bwt_j)
    BWT = np.mean(bwt_terms) if len(bwt_terms) > 0 else np.nan

    # Forward Transfer: FWT = mean(A[j-1,j] - chance) for j >= 1
    # CRITICAL: Only use measured cells (not NaN)
    fwt_terms = []
    if verbose:
        print("\nForward Transfer breakdown:")
    for j in range(1, T):
        pretrain = accuracy_matrix[j-1, j]  # Performance BEFORE training task j
        if not np.isnan(pretrain):
            fwt_j = pretrain - chance_level
            fwt_terms.append(fwt_j)
            if verbose:
                print(f"  FWT_{j+1} = {pretrain:.2f} - {chance_level:.2f} = {fwt_j:.2f}")
        else:
            if verbose:
                print(f"  FWT_{j+1} = NaN (not measured)")
    FWT = np.mean(fwt_terms) if len(fwt_terms) > 0 else np.nan

    if verbose:
        print(f"\nSummary:")
        print(f"  A_final = {A_final:.2f}%")
        print(f"  F_avg = {F_avg:.2f}")
        print(f"  BWT = {BWT:.2f} (should equal -F_avg)")
        print(f"  FWT = {FWT:.2f} (n={len(fwt_terms)} measured)")

    return ContinualLearningMetrics(
        A_final=A_final,
        F_avg=F_avg,
        BWT=BWT,
        FWT=FWT,
        n_fwt_measured=len(fwt_terms),
        accuracy_matrix=accuracy_matrix
    )


def validate_diagonal_peak(accuracy_matrix: np.ndarray, verbose: bool = False) -> Dict[str, float]:
    """
    Validate that peak performance occurs at diagonal.

    For disjoint task sequences (like Split-MNIST), we expect:
        max_i A[i,j] ≈ A[j,j]

    This justifies using diagonal-based forgetting instead of
    max-over-time forgetting.

    Args:
        accuracy_matrix: [T, T] accuracy matrix
        verbose: Print per-task results

    Returns:
        dict with 'max_deviation' and 'mean_deviation'
    """
    T = accuracy_matrix.shape[0]
    deviations = []

    if verbose:
        print("\nDiagonal peak validation:")
        print("Task | A[j,j] | max_i A[i,j] | δ_j")
        print("-" * 45)

    for j in range(T):
        diag = accuracy_matrix[j, j]
        max_val = np.nanmax(accuracy_matrix[:, j])
        deviation = max_val - diag

        deviations.append(deviation)

        if verbose:
            print(f"{j+1:4d} | {diag:6.2f} | {max_val:12.2f} | {deviation:4.2f}")

    max_dev = max(deviations)
    mean_dev = np.mean(deviations)

    if verbose:
        print(f"\nMax deviation: {max_dev:.2f}")
        print(f"Mean deviation: {mean_dev:.2f}")
        if max_dev < 0.5:
            print("✓ Diagonal peak validated (δ < 0.5)")
        else:
            print("⚠ Significant deviation - consider max-over-time forgetting")

    return {
        'max_deviation': max_dev,
        'mean_deviation': mean_dev,
        'deviations': deviations
    }


if __name__ == "__main__":
    # Example usage
    print("Testing metrics utilities...")

    # Test Φ approximation
    activations = torch.randn(4, 32, 16)
    phi = compute_phi_approximation(activations, method='variance')
    print(f"Φ (variance): {phi.mean().item():.4f}")

    phi = compute_phi_approximation(activations, method='correlation')
    print(f"Φ (correlation): {phi.mean().item():.4f}")

    # Test phase synchrony
    phases = torch.randn(32) * 2 * np.pi
    sync = compute_phase_synchrony(phases, mode='global')
    print(f"Global Synchrony: {sync.item():.4f}")

    # Test metrics logger
    logger = MetricsLogger()
    for i in range(10):
        logger.log({
            'loss': np.random.rand(),
            'phi': np.random.rand(),
            'free_energy': np.random.rand(),
        })

    summary = logger.get_summary(window=5)
    print(f"\nMetrics Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    # Test continual learning metrics
    print("\n" + "="*80)
    print("Testing Continual Learning Metrics")
    print("="*80)

    # Simulate accuracy matrix
    T = 5
    acc_matrix = np.array([
        [92.3, np.nan, np.nan, np.nan, np.nan],
        [85.1, 94.1, np.nan, np.nan, np.nan],
        [82.4, 89.3, 93.5, np.nan, np.nan],
        [80.7, 86.8, 90.2, 91.8, np.nan],
        [78.9, 84.5, 88.1, 89.3, 90.2]
    ])

    cl_metrics = compute_continual_learning_metrics(acc_matrix, verbose=True)

    print("\n" + "="*80)
    print("Validating diagonal peak assumption:")
    print("="*80)
    diagonal_validation = validate_diagonal_peak(acc_matrix, verbose=True)
