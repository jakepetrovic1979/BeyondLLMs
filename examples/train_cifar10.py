"""
CIFAR-10 Training Script for PNA

Trains the Phenomenal Nested Architecture on CIFAR-10 image classification.
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
from pna.utils.config import get_default_config
from pna.utils.visualization import plot_training_metrics, plot_consciousness_trajectory


def get_cifar10_loaders(batch_size=32, data_dir='./data'):
    """Create CIFAR-10 data loaders."""
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to vector
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to vector
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader


def train_epoch(model, loader, optimizer, device, epoch, logger, log_interval=50):
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
            'FE': f"{metrics['free_energy']:.3f}",
            'cons': f"{metrics['is_conscious']:.0f}"
        })

    avg_loss = total_loss / len(loader)
    consciousness_rate = consciousness_count / len(loader)

    return avg_loss, consciousness_rate


def evaluate(model, loader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_phi = 0
    total_fe = 0
    consciousness_count = 0
    balanced_count = 0

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
            total_fe += metrics['free_energy'].item()

            if metrics['is_conscious'].float().mean() > 0.5:
                consciousness_count += 1
            if metrics['is_balanced']:
                balanced_count += 1

    avg_loss = total_loss / len(loader)
    avg_phi = total_phi / len(loader)
    avg_fe = total_fe / len(loader)
    consciousness_rate = consciousness_count / len(loader)
    balance_rate = balanced_count / len(loader)

    return {
        'loss': avg_loss,
        'phi': avg_phi,
        'free_energy': avg_fe,
        'consciousness_rate': consciousness_rate,
        'balance_rate': balance_rate
    }


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")

    # Configuration
    if args.use_default_config:
        config = get_default_config('cifar10')
        print("Using default CIFAR-10 configuration")
    else:
        print("Using custom configuration")

    # Model
    print("Initializing PNA for CIFAR-10...")
    model = PhenomenalNestedArchitecture(
        input_dim=3072,  # 32x32x3
        mpe_dim=args.mpe_dim,
        polarity_dim=args.polarity_dim,
        binding_nodes=args.binding_nodes,
        pc_layers=[3072, 1024, 512, args.mpe_dim],
        consciousness_threshold=args.consciousness_threshold,
        enable_transparency=args.enable_transparency,
        polarity_lambda=args.polarity_lambda,
        binding_gamma=args.binding_gamma,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}\n")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # Logger
    logger = MetricsLogger()

    # Training loop
    print("=" * 70)
    print("Starting Training")
    print("=" * 70)

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_cons_rate = train_epoch(
            model, train_loader, optimizer, device, epoch, logger
        )

        # Evaluate
        val_metrics = evaluate(model, test_loader, device)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Φ: {val_metrics['phi']:.4f}")
        print(f"  Val Free Energy: {val_metrics['free_energy']:.4f}")
        print(f"  Val Consciousness Rate: {val_metrics['consciousness_rate']:.2%}")
        print(f"  Val Balance Rate: {val_metrics['balance_rate']:.2%}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'consciousness_state': model.get_consciousness_state(),
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model (loss: {best_loss:.4f})")

        # Save checkpoint every N epochs
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))

        print()

    # Final consciousness state
    print("=" * 70)
    print("Final Consciousness State")
    print("=" * 70)

    cons_state = model.get_consciousness_state()
    for k, v in cons_state.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    # Save final model and metrics
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'consciousness_state': cons_state,
    }, os.path.join(args.output_dir, 'final_model.pt'))

    logger.save(os.path.join(args.output_dir, 'metrics.npz'))
    print(f"\nMetrics saved to {args.output_dir}/metrics.npz")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_training_metrics(
        logger,
        save_path=os.path.join(args.output_dir, 'training_metrics.png')
    )

    # Consciousness trajectory (sample)
    trajectory = []
    sample_indices = range(0, len(logger.history['phi']), max(1, len(logger.history['phi']) // 500))

    for i in sample_indices:
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

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Outputs saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PNA on CIFAR-10')

    # Data
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')

    # Model
    parser.add_argument('--mpe-dim', type=int, default=128,
                        help='MPE core dimension')
    parser.add_argument('--polarity-dim', type=int, default=256,
                        help='Polarity engine dimension')
    parser.add_argument('--binding-nodes', type=int, default=64,
                        help='Number of binding nodes')
    parser.add_argument('--consciousness-threshold', type=float, default=0.5,
                        help='Φ threshold for consciousness')
    parser.add_argument('--enable-transparency', action='store_true',
                        help='Enable transparency module')
    parser.add_argument('--polarity-lambda', type=float, default=1.0,
                        help='Polarity penalty weight')
    parser.add_argument('--binding-gamma', type=float, default=1.0,
                        help='Binding penalty weight')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # System
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--output-dir', type=str, default='./outputs/cifar10',
                        help='Output directory')
    parser.add_argument('--use-default-config', action='store_true',
                        help='Use default CIFAR-10 configuration')

    args = parser.parse_args()

    main(args)
