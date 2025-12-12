"""
Basic PNA Example

Demonstrates basic usage of the Phenomenal Nested Architecture
with synthetic data.
"""

import torch
import torch.nn.functional as F
from pna.models.pna_model import PhenomenalNestedArchitecture
from pna.utils.metrics import MetricsLogger
from pna.utils.visualization import plot_training_metrics
from pna.utils.config import get_default_config


def main():
    print("=" * 60)
    print("Basic PNA Example - Synthetic Data")
    print("=" * 60)

    # Configuration
    config = get_default_config('mnist')
    device = torch.device(config.device)
    print(f"\nDevice: {device}")

    # Initialize PNA
    print("\nInitializing PNA...")
    pna = PhenomenalNestedArchitecture(
        input_dim=config.input_dim,
        mpe_dim=config.mpe_dim,
        polarity_dim=config.polarity_dim,
        binding_nodes=config.binding_nodes,
        pc_layers=config.pc_layers,
        consciousness_threshold=config.consciousness_threshold,
        enable_transparency=config.enable_transparency,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in pna.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(pna.parameters(), lr=config.learning_rate)

    # Metrics logger
    logger = MetricsLogger()

    # Training loop
    print("\n" + "=" * 60)
    print("Training on Synthetic Data")
    print("=" * 60)

    num_steps = 100
    batch_size = 32

    for step in range(num_steps):
        # Generate synthetic batch
        x = torch.randn(batch_size, config.input_dim).to(device)

        # Training step
        metrics = pna.train_step(x, optimizer, mode='balanced')

        # Log metrics
        logger.log(metrics)

        # Print progress
        if step % 10 == 0:
            print(f"\nStep {step}/{num_steps}")
            print(f"  Total Loss: {metrics['total_loss']:.6f}")
            print(f"  Free Energy: {metrics['free_energy']:.6f}")
            print(f"  Φ: {metrics['phi']:.4f}")
            print(f"  Is Conscious: {metrics['is_conscious']:.2f}")
            print(f"  Is Balanced: {metrics['is_balanced']:.2f}")

    # Final consciousness state
    print("\n" + "=" * 60)
    print("Final Consciousness State")
    print("=" * 60)

    cons_state = pna.get_consciousness_state()
    for k, v in cons_state.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Training Summary (last 50 steps)")
    print("=" * 60)

    summary = logger.get_summary(window=50)
    for k, v in summary.items():
        print(f"  {k}: {v:.6f}")

    # Save metrics
    logger.save('/tmp/pna_metrics.npz')
    print("\nMetrics saved to /tmp/pna_metrics.npz")

    # Visualize
    print("\nGenerating training plots...")
    plot_training_metrics(logger, save_path='/tmp/pna_training.png')

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
