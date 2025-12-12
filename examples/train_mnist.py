"""
MNIST Training Script for PNA

Trains the Phenomenal Nested Architecture on MNIST digit recognition,
tracking consciousness metrics throughout training.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
from tqdm import tqdm

from pna.models.pna_model import PhenomenalNestedArchitecture
from pna.utils.metrics import MetricsLogger, ConsciousnessMetrics
from pna.utils.config import PNAConfig
from pna.utils.visualization import plot_training_metrics, plot_consciousness_trajectory


def get_mnist_loaders(batch_size=32, data_dir='./data'):
    """Create MNIST data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to vector
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader


def train_epoch(model, loader, optimizer, device, epoch, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    consciousness_count = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')

    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device)

        # Training step
        metrics = model.train_step(data, optimizer, mode='balanced')

        # Log metrics
        logger.log(metrics)

        total_loss += metrics['total_loss']
        if metrics['is_conscious'] > 0.5:
            consciousness_count += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total_loss']:.4f}",
            'phi': f"{metrics['phi']:.3f}",
            'conscious': f"{metrics['is_conscious']:.0f}"
        })

    avg_loss = total_loss / len(loader)
    consciousness_rate = consciousness_count / len(loader)

    return avg_loss, consciousness_rate


def evaluate(model, loader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_phi = 0
    consciousness_count = 0

    with torch.no_grad():
        for data, target in tqdm(loader, desc='Evaluating'):
            data = data.to(device)

            # Forward pass
            output, metrics = model(data, return_all_metrics=False)

            # Reconstruction loss
            loss = F.mse_loss(output, data)
            total_loss += loss.item()

            # Consciousness metrics
            total_phi += metrics['phi'].mean().item()
            if metrics['is_conscious'].float().mean() > 0.5:
                consciousness_count += 1

    avg_loss = total_loss / len(loader)
    avg_phi = total_phi / len(loader)
    consciousness_rate = consciousness_count / len(loader)

    return avg_loss, avg_phi, consciousness_rate


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Data loaders
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")

    # Model
    print("Initializing PNA...")
    model = PhenomenalNestedArchitecture(
        input_dim=784,
        mpe_dim=args.mpe_dim,
        polarity_dim=args.polarity_dim,
        binding_nodes=args.binding_nodes,
        consciousness_threshold=args.consciousness_threshold,
        enable_transparency=args.enable_transparency,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}\n")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Logger
    logger = MetricsLogger()

    # Training loop
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_cons_rate = train_epoch(
            model, train_loader, optimizer, device, epoch, logger
        )

        # Evaluate
        val_loss, val_phi, val_cons_rate = evaluate(
            model, test_loader, device
        )

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Φ: {val_phi:.4f}")
        print(f"  Val Consciousness Rate: {val_cons_rate:.2%}")
        print(f"  Train Consciousness Rate: {train_cons_rate:.2%}")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model")

        print()

    # Final consciousness state
    print("=" * 60)
    print("Final Consciousness State")
    print("=" * 60)

    cons_state = model.get_consciousness_state()
    for k, v in cons_state.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, os.path.join(args.output_dir, 'final_model.pt'))

    # Save metrics
    logger.save(os.path.join(args.output_dir, 'metrics.npz'))
    print(f"\nMetrics saved to {args.output_dir}/metrics.npz")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_training_metrics(
        logger,
        save_path=os.path.join(args.output_dir, 'training_metrics.png')
    )

    # Consciousness trajectory
    trajectory = []
    for i in range(min(500, len(logger.history['phi']))):
        trajectory.append(
            ConsciousnessMetrics(
                phi=logger.history['phi'][i],
                synchrony=logger.history.get('synchrony', [0] * len(logger.history['phi']))[i],
                free_energy=logger.history['free_energy'][i],
                opacity=0.0,
                is_conscious=logger.history['is_conscious'][i] > 0.5,
                timestamp=i
            )
        )

    plot_consciousness_trajectory(
        trajectory,
        save_path=os.path.join(args.output_dir, 'consciousness_trajectory.png')
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PNA on MNIST')

    # Data
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')

    # Model
    parser.add_argument('--mpe-dim', type=int, default=64,
                        help='MPE core dimension')
    parser.add_argument('--polarity-dim', type=int, default=128,
                        help='Polarity engine dimension')
    parser.add_argument('--binding-nodes', type=int, default=32,
                        help='Number of binding nodes')
    parser.add_argument('--consciousness-threshold', type=float, default=0.5,
                        help='Φ threshold for consciousness')
    parser.add_argument('--enable-transparency', action='store_true',
                        help='Enable transparency module')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    # System
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--output-dir', type=str, default='./outputs/mnist',
                        help='Output directory')

    args = parser.parse_args()

    main(args)
