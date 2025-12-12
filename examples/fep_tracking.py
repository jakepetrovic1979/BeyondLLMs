"""
Free Energy Principle Tracking Demo

Demonstrates the MPE Core's FEP minimization with visualization
of the optimization trajectory.
"""

import torch
import matplotlib.pyplot as plt
from pna.core.mpe_core import MPECore


def main():
    print("=" * 60)
    print("Free Energy Principle - Tracking Demo")
    print("=" * 60)

    # Initialize MPE Core
    state_dim = 32
    obs_dim = 64

    mpe = MPECore(
        state_dim=state_dim,
        obs_dim=obs_dim,
        learning_rate=0.1,
        complexity_weight=0.01
    )

    print(f"\nMPE Core Configuration:")
    print(f"  State Dim: {state_dim}")
    print(f"  Observation Dim: {obs_dim}")

    # Create observation
    batch_size = 8
    obs = torch.randn(batch_size, obs_dim)

    print(f"\nMinimizing Free Energy...")

    # Minimize with trajectory tracking
    mu, precision, metrics = mpe.minimize_free_energy(
        obs,
        iterations=50,
        return_trajectory=True
    )

    print(f"\nFinal State:")
    print(f"  Free Energy: {metrics['free_energy'].mean().item():.6f}")
    print(f"  Complexity: {metrics['complexity'].mean().item():.6f}")
    print(f"  Accuracy: {metrics['accuracy'].mean().item():.6f}")
    print(f"  Prediction Error: {metrics['prediction_error'].item():.6f}")

    # MPE Properties
    print(f"\n" + "=" * 60)
    print("MPE Properties (Metzinger 2024)")
    print("=" * 60)

    props = mpe.get_mpe_properties()
    for k, v in props.items():
        print(f"  {k}: {v:.4f}")

    # Plot trajectory
    print(f"\nPlotting optimization trajectory...")

    trajectory = metrics['trajectory']
    iterations = [t['iteration'] for t in trajectory]
    free_energies = [t['free_energy'] for t in trajectory]
    complexities = [t['complexity'] for t in trajectory]
    accuracies = [t['accuracy'] for t in trajectory]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Free Energy
    axes[0].plot(iterations, free_energies, linewidth=2, color='purple')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Free Energy (F)')
    axes[0].set_title('Free Energy Minimization')
    axes[0].grid(True, alpha=0.3)

    # Complexity
    axes[1].plot(iterations, complexities, linewidth=2, color='orange')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Complexity (KL Divergence)')
    axes[1].set_title('Complexity Term')
    axes[1].grid(True, alpha=0.3)

    # Accuracy
    axes[2].plot(iterations, accuracies, linewidth=2, color='blue')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Accuracy (Log-Likelihood)')
    axes[2].set_title('Accuracy Term')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/fep_tracking.png', dpi=300, bbox_inches='tight')
    print(f"Saved to /tmp/fep_tracking.png")

    plt.show()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
