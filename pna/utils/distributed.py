"""
Distributed Training Support for PNA

Implements multi-GPU and multi-node training using PyTorch DDP
(DistributedDataParallel).
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from typing import Optional, Callable
import socket


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Initialize distributed training.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

    # Set device
    if backend == 'nccl':
        torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def is_main_process(rank: int = None) -> bool:
    """Check if current process is main process."""
    if rank is not None:
        return rank == 0
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def reduce_dict(input_dict: dict, average: bool = True):
    """
    Reduce dictionary of tensors across all processes.

    Args:
        input_dict: Dictionary of tensors to reduce
        average: Whether to average (True) or sum (False)

    Returns:
        Reduced dictionary
    """
    if not dist.is_initialized():
        return input_dict

    world_size = get_world_size()
    if world_size == 1:
        return input_dict

    with torch.no_grad():
        names = []
        values = []

        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict


class DistributedTrainer:
    """
    Distributed training wrapper for PNA.

    Handles multi-GPU training with DDP.
    """

    def __init__(
        self,
        model,
        rank: int,
        world_size: int,
        find_unused_parameters: bool = False
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')

        # Move model to device
        model = model.to(self.device)

        # Wrap with DDP
        self.model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=find_unused_parameters
        )

    def train_step(
        self,
        data,
        optimizer,
        mode: str = 'balanced'
    ) -> dict:
        """
        Distributed training step.

        Args:
            data: Input batch
            optimizer: Optimizer instance
            mode: Polarity mode

        Returns:
            Dictionary of metrics (reduced across processes)
        """
        data = data.to(self.device)

        # Forward and backward
        metrics = self.model.module.train_step(data, optimizer, mode)

        # Reduce metrics across processes
        metrics_tensor = torch.tensor(
            [metrics['total_loss'], metrics['phi'], metrics['free_energy']],
            device=self.device
        )

        dist.all_reduce(metrics_tensor)
        metrics_tensor /= self.world_size

        metrics['total_loss'] = metrics_tensor[0].item()
        metrics['phi'] = metrics_tensor[1].item()
        metrics['free_energy'] = metrics_tensor[2].item()

        return metrics

    def save_checkpoint(self, filepath: str, **kwargs):
        """Save checkpoint from main process only."""
        if is_main_process(self.rank):
            torch.save({
                'model_state_dict': self.model.module.state_dict(),
                **kwargs
            }, filepath)

    def load_checkpoint(self, filepath: str, strict: bool = True):
        """Load checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.module.load_state_dict(
            checkpoint['model_state_dict'],
            strict=strict
        )
        return checkpoint


def distributed_main(
    rank: int,
    world_size: int,
    train_fn: Callable,
    args
):
    """
    Main function for distributed training.

    Args:
        rank: Process rank
        world_size: Total number of processes
        train_fn: Training function
        args: Arguments object
    """
    # Setup
    setup_distributed(rank, world_size)

    # Run training
    try:
        train_fn(rank, world_size, args)
    finally:
        cleanup_distributed()


def launch_distributed(
    train_fn: Callable,
    args,
    num_gpus: Optional[int] = None
):
    """
    Launch distributed training.

    Args:
        train_fn: Training function with signature (rank, world_size, args)
        args: Arguments object
        num_gpus: Number of GPUs (None = all available)
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("No GPUs available. Running on CPU.")
        train_fn(0, 1, args)
        return

    world_size = num_gpus

    print(f"Launching distributed training on {world_size} GPUs...")

    mp.spawn(
        distributed_main,
        args=(world_size, train_fn, args),
        nprocs=world_size,
        join=True
    )


def get_distributed_sampler(dataset, shuffle: bool = True):
    """
    Create distributed sampler for dataset.

    Args:
        dataset: PyTorch dataset
        shuffle: Whether to shuffle

    Returns:
        DistributedSampler
    """
    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle
    )


def gather_tensors(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Gather tensor from all processes.

    Args:
        tensor: Tensor to gather [...]

    Returns:
        Gathered tensor [world_size, ...] (only on rank 0, None otherwise)
    """
    if not dist.is_initialized():
        return tensor.unsqueeze(0)

    world_size = get_world_size()

    # Create placeholder
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]

    # Gather
    dist.all_gather(tensor_list, tensor)

    if is_main_process():
        return torch.stack(tensor_list, dim=0)
    else:
        return None


def broadcast_object(obj, src: int = 0):
    """
    Broadcast Python object from source rank to all ranks.

    Args:
        obj: Object to broadcast
        src: Source rank

    Returns:
        Broadcasted object
    """
    if not dist.is_initialized():
        return obj

    if get_rank() == src:
        objects = [obj]
    else:
        objects = [None]

    dist.broadcast_object_list(objects, src=src)

    return objects[0]


def print_once(*args, **kwargs):
    """Print only from main process."""
    if is_main_process():
        print(*args, **kwargs)


def find_free_port() -> int:
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class DistributedMetrics:
    """
    Metrics aggregator for distributed training.

    Collects and reduces metrics across all processes.
    """

    def __init__(self):
        self.metrics = {}

    def update(self, metrics_dict: dict):
        """Update metrics from current batch."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []

            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()

            self.metrics[key].append(value)

    def reduce(self) -> dict:
        """Reduce metrics across all processes."""
        if not self.metrics:
            return {}

        # Compute local averages
        local_avg = {
            key: sum(values) / len(values)
            for key, values in self.metrics.items()
        }

        # Convert to tensor
        keys = sorted(local_avg.keys())
        values = torch.tensor(
            [local_avg[k] for k in keys],
            dtype=torch.float32
        )

        # Reduce across processes
        if dist.is_initialized():
            dist.all_reduce(values)
            values /= get_world_size()

        # Convert back to dict
        reduced = {k: v.item() for k, v in zip(keys, values)}

        return reduced

    def reset(self):
        """Reset metrics."""
        self.metrics = {}


if __name__ == "__main__":
    print("Distributed Training Module for PNA")
    print("\nSupports:")
    print("  - Multi-GPU training with DDP")
    print("  - Distributed metrics aggregation")
    print("  - Checkpointing and synchronization")
    print("\nUsage:")
    print("  launch_distributed(train_fn, args, num_gpus=4)")
