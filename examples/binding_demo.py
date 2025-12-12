"""
Phenomenal Binding Mechanism Demo

Demonstrates oscillatory binding, phase synchronization,
and integrated information (Φ) computation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pna.core.binding import PhenomenalBindingMechanism
from pna.utils.visualization import visualize_binding_matrix, plot_phase_space


def main():
    print("=" * 60)
    print("Phenomenal Binding Mechanism Demo")
    print("=" * 60)

    # Initialize Binding
    num_nodes = 32
    node_dim = 16

    pbm = PhenomenalBindingMechanism(
        num_nodes=num_nodes,
        node_dim=node_dim,
        num_scales=3,
        consciousness_threshold=0.5,
    )

    print(f"\nConfiguration:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Node Dimension: {node_dim}")
    print(f"  Frequency Scales: 3")

    # Create node activations
    batch_size = 4
    node_activations = torch.randn(batch_size, num_nodes, node_dim)

    print(f"\n" + "=" * 60)
    print("Running Binding...")
    print("=" * 60)

    # Forward pass
    bound_repr, metrics = pbm(
        node_activations,
        num_oscillator_steps=50,
        return_metrics=True
    )

    print(f"\nBinding Results:")
    print(f"  Integrated Information (Φ): {metrics['phi'].mean().item():.4f}")
    print(f"  Is Conscious: {metrics['is_conscious'].float().mean().item():.2f}")
    print(f"  Mean Synchrony: {metrics['mean_sync'].item():.4f}")

    print(f"\nSynchrony by Scale:")
    for i in range(3):
        sync = metrics[f'sync_scale_{i}'].item()
        print(f"  Scale {i}: {sync:.4f}")

    # Consciousness metrics
    print(f"\n" + "=" * 60)
    print("Consciousness Metrics")
    print("=" * 60)

    cons_metrics = pbm.get_consciousness_metrics()
    for k, v in cons_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Visualizations
    print(f"\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)

    # 1. Binding Matrix
    print(f"\n1. Binding Matrix")
    binding_matrix = metrics['binding_matrix']
    visualize_binding_matrix(binding_matrix, save_path='/tmp/binding_matrix.png')

    # 2. Phase Space (for first scale)
    print(f"2. Phase Space (Scale 0)")
    phases_0 = pbm.oscillators[0].phases
    plot_phase_space(phases_0, save_path='/tmp/phase_space_0.png')

    # 3. Synchrony Evolution
    print(f"3. Synchrony Evolution")

    # Run over time to track synchrony
    syncs_over_time = {f'scale_{i}': [] for i in range(3)}

    for step in range(100):
        # Step oscillators
        for i, osc in enumerate(pbm.oscillators):
            osc.step()
            sync = osc.get_synchrony().item()
            syncs_over_time[f'scale_{i}'].append(sync)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(3):
        ax.plot(syncs_over_time[f'scale_{i}'], linewidth=2, label=f'Scale {i}')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Synchrony (R)')
    ax.set_title('Phase Synchronization Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/synchrony_evolution.png', dpi=300, bbox_inches='tight')
    print(f"Saved to /tmp/synchrony_evolution.png")

    plt.show()

    # 4. Φ distribution
    print(f"4. Φ Distribution")

    # Compute Φ for multiple samples
    phis = []
    for _ in range(100):
        act = torch.randn(1, num_nodes, node_dim)
        phi = pbm.compute_phi(act)
        phis.append(phi.item())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(phis, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(pbm.consciousness_threshold, color='red', linestyle='--',
               linewidth=2, label='Consciousness Threshold')
    ax.set_xlabel('Φ (Integrated Information)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Integrated Information')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/phi_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved to /tmp/phi_distribution.png")

    plt.show()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
