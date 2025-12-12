"""
Distributed Training Script for PNA

Example of multi-GPU training using PyTorch DDP.

Usage:
    # Single node, multiple GPUs
    python train_distributed.py --num-gpus 4

    # Multi-node (on each node)
    python train_distributed.py --num-gpus 4 --num-nodes 2 --node-rank 0
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
from tqdm import tqdm

from pna.models.pna_model import PhenomenalNestedArchitecture
from pna.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    DistributedTrainer,
    get_distributed_sampler,
    is_main_process,
    print_once,
    DistributedMetrics
)
from pna.utils.metrics import MetricsLogger


def get_data_loaders(rank, world_size, args):
    """Create distributed data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=is_main_process(rank),  # Only download from main process
        transform=transform
    )

    # Wait for download to complete
    if torch.cuda.is_available():
        torch.distributed.barrier()

    # Create distributed sampler
    train_sampler = get_distributed_sampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_loader


def train_epoch(trainer, train_loader, optimizer, epoch, rank):
    """Train for one epoch."""
    trainer.model.train()
    metrics_agg = DistributedMetrics()

    # Only show progress bar on main process
    if is_main_process(rank):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        iterator = pbar
    else:
        iterator = train_loader

    for batch_idx, (data, target) in enumerate(iterator):
        # Distributed training step
        metrics = trainer.train_step(data, optimizer, mode='balanced')

        # Update metrics
        metrics_agg.update(metrics)

        # Update progress bar (main process only)
        if is_main_process(rank) and batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'phi': f"{metrics['phi']:.3f}",
                'conscious': f"{metrics.get('is_conscious', 0):.0f}"
            })

    # Reduce metrics across all processes
    epoch_metrics = metrics_agg.reduce()

    return epoch_metrics


def train_distributed(rank, world_size, args):
    """Main distributed training function."""
    # Setup
    setup_distributed(rank, world_size, backend='nccl')

    print_once(f"Training on {world_size} GPUs")

    # Create output directory (main process only)
    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)

    # Data loaders
    train_loader = get_data_loaders(rank, world_size, args)

    print_once(f"Training samples: {len(train_loader.dataset)}")

    # Model
    print_once("Initializing PNA...")
    model = PhenomenalNestedArchitecture(
        input_dim=784,
        mpe_dim=args.mpe_dim,
        polarity_dim=args.polarity_dim,
        binding_nodes=args.binding_nodes,
        consciousness_threshold=args.consciousness_threshold,
    )

    # Distributed trainer
    trainer = DistributedTrainer(
        model,
        rank=rank,
        world_size=world_size,
        find_unused_parameters=False
    )

    print_once(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(
        trainer.model.parameters(),
        lr=args.lr
    )

    # Logger (main process only)
    if is_main_process(rank):
        logger = MetricsLogger()
    else:
        logger = None

    # Training loop
    print_once("=" * 60)
    print_once("Starting Distributed Training")
    print_once("=" * 60)

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Set epoch for distributed sampler
        train_loader.sampler.set_epoch(epoch)

        # Train
        epoch_metrics = train_epoch(
            trainer, train_loader, optimizer, epoch, rank
        )

        # Log metrics (main process only)
        if is_main_process(rank):
            logger.log(epoch_metrics)

            print(f"\nEpoch {epoch} Summary:")
            print(f"  Loss: {epoch_metrics['total_loss']:.4f}")
            print(f"  Φ: {epoch_metrics.get('phi', 0):.4f}")
            print(f"  Free Energy: {epoch_metrics.get('free_energy', 0):.4f}")

            # Save best model
            if epoch_metrics['total_loss'] < best_loss:
                best_loss = epoch_metrics['total_loss']
                trainer.save_checkpoint(
                    os.path.join(args.output_dir, 'best_model.pt'),
                    epoch=epoch,
                    loss=best_loss,
                    optimizer_state_dict=optimizer.state_dict()
                )
                print(f"  ✓ Saved best model")

        # Synchronize all processes
        torch.distributed.barrier()

    # Final save (main process only)
    if is_main_process(rank):
        trainer.save_checkpoint(
            os.path.join(args.output_dir, 'final_model.pt'),
            epoch=args.epochs,
            optimizer_state_dict=optimizer.state_dict()
        )

        logger.save(os.path.join(args.output_dir, 'metrics.npz'))

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best loss: {best_loss:.4f}")
        print("=" * 60)

    # Cleanup
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Distributed PNA Training')

    # Distributed
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs (None = all available)')
    parser.add_argument('--num-nodes', type=int, default=1,
                        help='Number of nodes')
    parser.add_argument('--node-rank', type=int, default=0,
                        help='Rank of this node')
    parser.add_argument('--dist-url', type=str, default='env://',
                        help='URL for distributed training')

    # Data
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size per GPU')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Data loader workers per GPU')

    # Model
    parser.add_argument('--mpe-dim', type=int, default=64,
                        help='MPE core dimension')
    parser.add_argument('--polarity-dim', type=int, default=128,
                        help='Polarity engine dimension')
    parser.add_argument('--binding-nodes', type=int, default=32,
                        help='Number of binding nodes')
    parser.add_argument('--consciousness-threshold', type=float, default=0.5,
                        help='Φ threshold')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    # Output
    parser.add_argument('--output-dir', type=str, default='./outputs/distributed',
                        help='Output directory')

    args = parser.parse_args()

    # Determine number of GPUs
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()

    if args.num_gpus == 0:
        print("ERROR: No GPUs available for distributed training")
        return

    # Launch distributed training
    from pna.utils.distributed import launch_distributed
    launch_distributed(train_distributed, args, num_gpus=args.num_gpus)


if __name__ == "__main__":
    main()
