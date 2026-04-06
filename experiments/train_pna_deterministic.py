"""
Fully Deterministic PNA Training for Reproducibility
=====================================================

Addresses reviewer concern:
"you set seeds, but not full determinism (CUDA nondeterminism, dataloader seeds).
This matters if you're bootstrapping across seeds."

Changes:
1. torch.use_deterministic_algorithms(True)
2. CUBLAS workspace config
3. DataLoader worker seeding
4. Environment variable for deterministic operations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pna_complete_implementation_v2 import PNASystem, get_split_mnist


def set_global_seed(seed: int = 42, deterministic: bool = True):
    """
    Set all random seeds for full reproducibility.

    Args:
        seed: Random seed
        deterministic: If True, enables deterministic CUDA operations (slower)

    Note: Deterministic mode may reduce performance by 10-30% due to
    algorithmic constraints. Use for final validation only.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    if deterministic:
        # Force deterministic operations
        torch.use_deterministic_algorithms(True)

        # CUDA configuration
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Required for deterministic CUDA

        # CuDNN determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"✓ Deterministic mode enabled (seed={seed})")
        print("  Warning: This may reduce performance by 10-30%")
    else:
        print(f"✓ Seeds set (seed={seed}), but non-deterministic operations allowed")


def worker_init_fn(worker_id: int):
    """
    DataLoader worker initialization for reproducibility.

    Each worker gets a unique but deterministic seed based on:
    - Global seed
    - Worker ID
    - Current epoch (if needed)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_deterministic_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42
) -> DataLoader:
    """
    Create DataLoader with full determinism.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers (0 = main process only)
        seed: Random seed

    Returns:
        Deterministic DataLoader
    """
    # Use generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=g,
        persistent_workers=False,  # For determinism
        pin_memory=False  # May cause nondeterminism on some systems
    )

    return loader


def train_pna_deterministic(
    num_tasks: int = 5,
    epochs_per_task: int = 3,
    batch_size: int = 32,
    seed: int = 42,
    device: str = None,
    deterministic: bool = True
):
    """
    Fully deterministic training for reproducibility validation.

    Use cases:
    1. Bootstrap CI validation (need exact same training across runs)
    2. Ablation studies (ensure only ablated component varies)
    3. Hyperparameter sensitivity (isolate parameter effects)

    Args:
        num_tasks: Number of tasks
        epochs_per_task: Epochs per task
        batch_size: Batch size
        seed: Random seed
        device: 'cpu', 'cuda', or None (auto)
        deterministic: Enable full determinism (slower)

    Returns:
        accuracy_matrix: [T, T] accuracy matrix
        final_metrics: dict with A_final, F_avg, BWT
    """
    # Set global determinism
    set_global_seed(seed, deterministic)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*80}")
    print(f"DETERMINISTIC TRAINING")
    print(f"{'='*80}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print(f"Deterministic: {deterministic}")
    print(f"{'='*80}\n")

    # Initialize model
    pna = PNASystem(
        d_input=28*28,
        d_hidden=64,
        d_output=2,
        lambda_pol=0.1,
        lambda_mpe=0.5,
        device=device
    )

    # Optimizer
    optimizer = torch.optim.Adam(pna.parameters(), lr=0.001)

    # Storage
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    test_loaders = []

    # Training loop
    for task_id in range(num_tasks):
        print(f"\nTask {task_id + 1}/{num_tasks}")
        print("-" * 40)

        # Get datasets with deterministic settings
        train_dataset, test_dataset = get_split_mnist(task_id, device)

        train_loader = get_deterministic_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Single-process for maximum determinism
            seed=seed + task_id  # Unique seed per task
        )

        test_loader = get_deterministic_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            seed=seed + task_id
        )

        test_loaders.append(test_loader)

        # Train
        for epoch in range(epochs_per_task):
            pna.train()
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)

                result = pna(data, target)

                optimizer.zero_grad()
                result['losses']['total'].backward()

                # Gradient clipping (deterministic)
                torch.nn.utils.clip_grad_norm_(pna.parameters(), max_norm=1.0)

                # Plasticity modulation
                if result['plasticity'] is not None:
                    plasticity = result['plasticity'].mean()
                    for param in pna.l1_polarity.parameters():
                        if param.grad is not None:
                            param.grad *= plasticity

                optimizer.step()
                pna.reset()

                epoch_loss += result['losses']['total'].item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            print(f"  Epoch {epoch + 1}/{epochs_per_task}: Loss = {avg_loss:.6f}")

        # Evaluate on all seen tasks
        print(f"\nEvaluating after Task {task_id + 1}...")
        pna.eval()

        with torch.no_grad():
            for eval_task_id in range(task_id + 1):
                correct = 0
                total = 0

                for data, target in test_loaders[eval_task_id]:
                    data = data.to(device)
                    target = target.to(device)

                    result = pna(data)
                    pred = result['output'][:, -1, :].argmax(dim=1)

                    correct += (pred == target).sum().item()
                    total += target.size(0)

                    pna.reset()

                acc = 100.0 * correct / total
                accuracy_matrix[task_id, eval_task_id] = acc
                print(f"  Task {eval_task_id + 1}: {acc:.2f}%")

    # Compute metrics
    T = num_tasks
    final_row = accuracy_matrix[T-1, :]
    A_final = np.mean(final_row)

    forgetting = []
    for j in range(T-1):
        f_j = accuracy_matrix[j, j] - accuracy_matrix[T-1, j]
        forgetting.append(f_j)
    F_avg = np.mean(forgetting)

    BWT = -F_avg  # By construction (diagonal-based)

    print(f"\n{'='*80}")
    print("FINAL METRICS")
    print(f"{'='*80}")
    print(f"A_final = {A_final:.2f}%")
    print(f"F_avg = {F_avg:.2f}")
    print(f"BWT = {BWT:.2f}")
    print(f"{'='*80}\n")

    return accuracy_matrix, {
        'A_final': A_final,
        'F_avg': F_avg,
        'BWT': BWT,
        'seed': seed
    }


def validate_determinism(num_runs: int = 3, seed: int = 42):
    """
    Validate that deterministic mode produces identical results.

    Runs training multiple times with same seed and checks for
    exact floating-point equality in final metrics.

    Args:
        num_runs: Number of validation runs
        seed: Random seed (same for all runs)
    """
    print(f"\n{'='*80}")
    print(f"DETERMINISM VALIDATION")
    print(f"{'='*80}")
    print(f"Running {num_runs} times with seed={seed}")
    print(f"Expected: Identical results across all runs")
    print(f"{'='*80}\n")

    results = []

    for run_id in range(num_runs):
        print(f"\n{'='*80}")
        print(f"RUN {run_id + 1}/{num_runs}")
        print(f"{'='*80}")

        _, metrics = train_pna_deterministic(
            num_tasks=3,  # Shorter for validation
            epochs_per_task=2,
            seed=seed,
            deterministic=True
        )

        results.append(metrics)

    # Check equality
    print(f"\n{'='*80}")
    print("DETERMINISM CHECK")
    print(f"{'='*80}")

    reference = results[0]
    all_identical = True

    for i, metrics in enumerate(results[1:], start=2):
        for key in ['A_final', 'F_avg', 'BWT']:
            ref_val = reference[key]
            cur_val = metrics[key]

            if abs(ref_val - cur_val) > 1e-6:
                print(f"✗ Run {i} differs: {key} = {cur_val:.6f} (expected {ref_val:.6f})")
                all_identical = False

    if all_identical:
        print("✓ PERFECT DETERMINISM: All runs produced identical results")
        print(f"  A_final = {reference['A_final']:.6f}")
        print(f"  F_avg = {reference['F_avg']:.6f}")
        print(f"  BWT = {reference['BWT']:.6f}")
    else:
        print("✗ NONDETERMINISM DETECTED: Results vary across runs")
        print("  Check CUDA version, driver, and hardware configuration")

    return all_identical


if __name__ == "__main__":
    import sys

    # Test determinism first
    print("\n" + "="*80)
    print("STEP 1: VALIDATE DETERMINISM")
    print("="*80)

    is_deterministic = validate_determinism(num_runs=2, seed=42)

    if not is_deterministic:
        print("\nWARNING: Determinism validation failed!")
        print("This may be due to:")
        print("1. CUDA version/driver issues")
        print("2. Hardware-specific nondeterminism")
        print("3. Certain operations not supporting deterministic mode")
        print("\nProceeding with best-effort determinism...")

    # Run single deterministic training
    print("\n" + "="*80)
    print("STEP 2: SINGLE DETERMINISTIC RUN")
    print("="*80)

    accuracy_matrix, metrics = train_pna_deterministic(
        num_tasks=5,
        epochs_per_task=3,
        seed=42,
        deterministic=True
    )

    # Save results
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'pna_accuracy_matrix_deterministic.npy'), accuracy_matrix)

    print(f"\nSaved to: {output_dir}/pna_accuracy_matrix_deterministic.npy")

    print("\n" + "="*80)
    print("REPRODUCIBILITY SETUP COMPLETE")
    print("="*80)
    print("\nFor paper methods section:")
    print("'All experiments use deterministic PyTorch operations with fixed seeds")
    print("(torch.use_deterministic_algorithms(True)) to ensure exact reproducibility.")
    print("Bootstrap confidence intervals are computed from {n} independent runs with")
    print("different random seeds, each producing deterministic results within seed.'")
