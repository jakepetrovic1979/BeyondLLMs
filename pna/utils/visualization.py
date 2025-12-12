"""Visualization utilities for PNA."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pna.utils.metrics import ConsciousnessMetrics


def plot_consciousness_trajectory(
    trajectory: List[ConsciousnessMetrics],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Plot consciousness metrics over time.

    Args:
        trajectory: List of ConsciousnessMetrics
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    timestamps = [m.timestamp for m in trajectory]
    phis = [m.phi for m in trajectory]
    syncs = [m.synchrony for m in trajectory]
    fes = [m.free_energy for m in trajectory]
    opacities = [m.opacity for m in trajectory]

    # Φ
    axes[0, 0].plot(timestamps, phis, linewidth=2, color='purple')
    axes[0, 0].axhline(trajectory[0].phi if trajectory else 0.5,
                       color='red', linestyle='--', alpha=0.5, label='Threshold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Φ (Integrated Information)')
    axes[0, 0].set_title('Integrated Information (Φ)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Synchrony
    axes[0, 1].plot(timestamps, syncs, linewidth=2, color='blue')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Synchrony')
    axes[0, 1].set_title('Phase Synchronization')
    axes[0, 1].grid(True, alpha=0.3)

    # Free Energy
    axes[1, 0].plot(timestamps, fes, linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Free Energy')
    axes[1, 0].set_title('Free Energy (Surprise)')
    axes[1, 0].grid(True, alpha=0.3)

    # Opacity
    axes[1, 1].plot(timestamps, opacities, linewidth=2, color='green')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Opacity (Ω)')
    axes[1, 1].set_title('Representational Opacity')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def visualize_binding_matrix(
    binding_matrix: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    cmap: str = 'viridis'
):
    """
    Visualize phenomenal binding matrix.

    Args:
        binding_matrix: Binding strength matrix [num_nodes, num_nodes]
        save_path: Optional path to save figure
        figsize: Figure size
        cmap: Colormap
    """
    if torch.is_tensor(binding_matrix):
        binding_matrix = binding_matrix.cpu().numpy()

    plt.figure(figsize=figsize)

    sns.heatmap(
        binding_matrix,
        cmap=cmap,
        square=True,
        cbar_kws={'label': 'Binding Strength'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.title('Phenomenal Binding Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_polarity_balance(
    gen_rates: List[float],
    rad_rates: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot polarity engine balance over time.

    Args:
        gen_rates: Generative rates (dG/dt)
        rad_rates: Radiative rates (dR/dt)
        save_path: Optional path to save
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    timestamps = range(len(gen_rates))

    # Rates
    axes[0].plot(timestamps, gen_rates, label='Generative (dG/dt)', linewidth=2, color='blue')
    axes[0].plot(timestamps, rad_rates, label='Radiative (dR/dt)', linewidth=2, color='red')
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Rate')
    axes[0].set_title('Polarity Rates')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Balance (sum)
    balance = [g + r for g, r in zip(gen_rates, rad_rates)]
    axes[1].plot(timestamps, balance, linewidth=2, color='purple')
    axes[1].axhline(0, color='green', linestyle='--', alpha=0.5, label='Perfect Balance')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('dG/dt + dR/dt')
    axes[1].set_title('Polarity Balance (Ψ)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_pc_hierarchy(
    layer_errors: List[float],
    layer_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
):
    """
    Plot predictive coding hierarchy errors.

    Args:
        layer_errors: Errors at each layer
        layer_names: Optional layer names
        save_path: Optional save path
        figsize: Figure size
    """
    if layer_names is None:
        layer_names = [f"Layer {i}" for i in range(len(layer_errors))]

    plt.figure(figsize=figsize)

    x = range(len(layer_errors))
    plt.bar(x, layer_errors, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xticks(x, layer_names, rotation=45)
    plt.ylabel('Prediction Error')
    plt.title('Predictive Coding Hierarchy Errors')
    plt.grid(True, axis='y', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_training_metrics(
    metrics_logger,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10)
):
    """
    Plot comprehensive training metrics from MetricsLogger.

    Args:
        metrics_logger: MetricsLogger instance
        save_path: Optional save path
        figsize: Figure size
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()

    plot_configs = [
        ('loss', 'Training Loss', 'orange'),
        ('phi', 'Integrated Information (Φ)', 'purple'),
        ('free_energy', 'Free Energy', 'blue'),
        ('polarity_penalty', 'Polarity Penalty (Ψ)', 'red'),
        ('synchrony', 'Phase Synchrony', 'green'),
        ('is_conscious', 'Consciousness (binary)', 'magenta'),
    ]

    for idx, (key, title, color) in enumerate(plot_configs):
        if key in metrics_logger.history and metrics_logger.history[key]:
            values = metrics_logger.history[key]
            axes[idx].plot(values, linewidth=2, color=color, alpha=0.7)
            axes[idx].set_xlabel('Step')
            axes[idx].set_ylabel(title)
            axes[idx].set_title(title)
            axes[idx].grid(True, alpha=0.3)

            # Add moving average
            if len(values) > 20:
                window = min(50, len(values) // 10)
                ma = np.convolve(values, np.ones(window)/window, mode='valid')
                axes[idx].plot(range(window-1, len(values)), ma,
                             linewidth=2, color='black', linestyle='--',
                             alpha=0.5, label=f'MA({window})')
                axes[idx].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_phase_space(
    phases: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8)
):
    """
    Plot oscillator phases in phase space.

    Args:
        phases: Oscillator phases [num_oscillators]
        save_path: Optional save path
        figsize: Figure size
    """
    if torch.is_tensor(phases):
        phases = phases.cpu().numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')

    # Convert to [0, 2π]
    phases_norm = (phases + np.pi) % (2 * np.pi)

    # Plot as points on unit circle
    r = np.ones_like(phases_norm)
    ax.scatter(phases_norm, r, c=range(len(phases)), cmap='hsv', s=100, alpha=0.7)

    # Plot mean phase (order parameter)
    mean_phase = np.angle(np.mean(np.exp(1j * phases)))
    ax.arrow(mean_phase, 0, 0, 0.8, head_width=0.1, head_length=0.1,
             fc='red', ec='red', linewidth=2, alpha=0.8, label='Mean Phase')

    ax.set_ylim(0, 1.2)
    ax.set_title('Oscillator Phase Distribution', pad=20)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Testing visualization utilities...")

    # Create synthetic trajectory
    trajectory = [
        ConsciousnessMetrics(
            phi=0.3 + 0.05 * np.sin(i * 0.1),
            synchrony=0.5 + 0.1 * np.cos(i * 0.15),
            free_energy=1.0 - 0.01 * i,
            opacity=0.8 + 0.05 * np.sin(i * 0.2),
            is_conscious=True,
            timestamp=i
        )
        for i in range(100)
    ]

    plot_consciousness_trajectory(trajectory, save_path='/tmp/consciousness_traj.png')

    # Binding matrix
    binding_matrix = torch.rand(32, 32)
    binding_matrix = (binding_matrix + binding_matrix.T) / 2  # Symmetric
    visualize_binding_matrix(binding_matrix, save_path='/tmp/binding_matrix.png')

    # Polarity balance
    gen_rates = [np.sin(i * 0.1) for i in range(100)]
    rad_rates = [-np.sin(i * 0.1 + 0.5) for i in range(100)]
    plot_polarity_balance(gen_rates, rad_rates, save_path='/tmp/polarity.png')

    print("Visualizations created successfully!")
