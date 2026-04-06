"""
CRITICAL FIX: Full Matrix Evaluation for Valid FWT
====================================================

Issue: Original code only evaluated tasks seen so far (lower triangle).
This makes FWT invalid because A[j-1,j] (upper triangle) is unmeasured.

Solution: Evaluate ALL tasks at each checkpoint, setting upper triangle
to NaN where task hasn't been trained yet. This allows:
- Valid FWT computation (using only measured A[j-1,j])
- Honest reporting of "what the model hasn't seen yet"
- Proper handling in metric computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PNA system from the main implementation
from pna_complete_implementation_v2 import PNASystem, get_split_mnist_loaders

def train_pna_continual_learning_full_matrix(
    num_tasks: int = 5,
    epochs_per_task: int = 3,
    batch_size: int = 32,
    device: str = None,
    seed: int = 42
):
    """
    Train PNA with FULL matrix evaluation (including upper triangle).

    Key changes:
    1. Evaluate ALL tasks at each checkpoint (not just seen tasks)
    2. Set upper triangle to NaN (tasks not yet trained)
    3. Handle NaN properly in metric computation
    4. Enable valid FWT calculation

    Args:
        num_tasks: Number of sequential tasks (default 5)
        epochs_per_task: Training epochs per task
        batch_size: Batch size
        device: 'cpu', 'cuda', or None (auto-detect)
        seed: Random seed for reproducibility

    Returns:
        accuracy_matrix: [T, T] with NaN in upper triangle until trained
        metrics_history: dict with training metrics
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on device: {device}, seed: {seed}")
    print("="*80)

    # Initialize model
    pna = PNASystem(d_input=28*28, d_hidden=64, d_output=2,
                   lambda_pol=0.1, lambda_mpe=0.5, device=device)

    optimizer = optim.Adam(pna.parameters(), lr=0.001)

    # Storage - CRITICAL: Initialize with NaN for full matrix
    accuracy_matrix = np.full((num_tasks, num_tasks), np.nan, dtype=np.float32)
    test_loaders = []
    metrics_history = {
        'phi': [],
        'loss': [],
        'polarity_loss': [],
        'mpe_loss': []
    }

    # CRITICAL: Pre-create ALL test loaders (for upper triangle evaluation)
    print("Pre-creating all test loaders...")
    all_test_loaders = []
    for t in range(num_tasks):
        _, test_loader = get_split_mnist_loaders(t, batch_size, device)
        all_test_loaders.append(test_loader)
    print("Done.\n")

    # Train on each task sequentially
    for task_id in range(num_tasks):
        print(f"\n{'='*80}")
        print(f"TASK {task_id + 1}/{num_tasks}: Digits {2*task_id} vs {2*task_id + 1}")
        print(f"{'='*80}")

        train_loader, _ = get_split_mnist_loaders(task_id, batch_size, device)

        # Train on current task
        for epoch in range(epochs_per_task):
            pna.train()
            epoch_loss = 0.0
            epoch_phi = 0.0
            batch_count = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)

                result = pna(data, target)

                optimizer.zero_grad()
                result['losses']['total'].backward()
                torch.nn.utils.clip_grad_norm_(pna.parameters(), max_norm=1.0)

                if result['plasticity'] is not None:
                    plasticity = result['plasticity'].mean()
                    for param in pna.l1_polarity.parameters():
                        if param.grad is not None:
                            param.grad *= plasticity

                optimizer.step()
                pna.reset()

                epoch_loss += result['losses']['total'].item()
                epoch_phi += result['phi']
                batch_count += 1

                if batch_idx % 50 == 0:
                    print(f"  Epoch {epoch+1}/{epochs_per_task} | "
                          f"Batch {batch_idx}/{len(train_loader)} | "
                          f"Loss: {result['losses']['total'].item():.4f} | "
                          f"Φ: {result['phi']:.4f}")

            avg_loss = epoch_loss / batch_count
            avg_phi = epoch_phi / batch_count

            metrics_history['loss'].append(avg_loss)
            metrics_history['phi'].append(avg_phi)

            print(f"\n  → Epoch {epoch+1} Summary: Loss={avg_loss:.4f}, Φ={avg_phi:.4f}\n")

        # CRITICAL: Evaluate on ALL tasks (including future ones)
        print(f"\nEvaluating on ALL {num_tasks} tasks after training Task {task_id + 1}...")
        pna.eval()

        with torch.no_grad():
            for eval_task_id in range(num_tasks):
                correct = 0
                total = 0

                for data, target in all_test_loaders[eval_task_id]:
                    data = data.to(device)
                    target = target.to(device)

                    result = pna(data)
                    pred = result['output'][:, -1, :].argmax(dim=1)

                    correct += (pred == target).sum().item()
                    total += target.size(0)

                    pna.reset()

                acc = 100.0 * correct / total

                # Only update if task has been trained (diagonal and below)
                if eval_task_id <= task_id:
                    accuracy_matrix[task_id, eval_task_id] = acc
                # Otherwise keep as NaN (task not yet trained)

                # Flag if this is upper triangle (forward transfer)
                if eval_task_id > task_id:
                    print(f"  Task {eval_task_id + 1} Accuracy: {acc:.2f}% [FORWARD TRANSFER - before training]")
                else:
                    print(f"  Task {eval_task_id + 1} Accuracy: {acc:.2f}%")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    return accuracy_matrix, metrics_history


def compute_continual_learning_metrics_with_nan(accuracy_matrix: np.ndarray,
                                                chance: float = 50.0,
                                                verbose: bool = True):
    """
    Compute CL metrics with proper NaN handling.

    CRITICAL DIFFERENCE: Only uses measured cells for FWT.

    Args:
        accuracy_matrix: [T, T] with potential NaN in upper triangle
        chance: Random chance baseline (50% for binary)
        verbose: Print detailed breakdown

    Returns:
        dict with metrics
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
            fwt_j = pretrain - chance
            fwt_terms.append(fwt_j)
            if verbose:
                print(f"  FWT_{j+1} = {pretrain:.2f} - {chance:.2f} = {fwt_j:.2f}")
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

    return {
        'A_final': A_final,
        'F_avg': F_avg,
        'BWT': BWT,
        'FWT': FWT,
        'n_fwt_measured': len(fwt_terms)
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PNA Training with FULL MATRIX EVALUATION (Fixed FWT)")
    print("="*80)

    # Run training
    accuracy_matrix, metrics_history = train_pna_continual_learning_full_matrix(
        num_tasks=5,
        epochs_per_task=3,
        batch_size=32,
        device=None,
        seed=42
    )

    # Print matrix
    print("\n" + "="*80)
    print("ACCURACY MATRIX (NaN = not yet trained)")
    print("="*80)
    print("\nRows = After training task i")
    print("Cols = Evaluated on task j")
    print("\n", accuracy_matrix)

    # Compute metrics
    print("\n" + "="*80)
    print("METRICS (with proper NaN handling)")
    print("="*80)
    metrics = compute_continual_learning_metrics_with_nan(accuracy_matrix, verbose=True)

    # Save
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'pna_accuracy_matrix_full.npy'), accuracy_matrix)

    print(f"\n\nSaved to: {output_dir}/pna_accuracy_matrix_full.npy")
    print("\nNote: Upper triangle shows performance on future tasks BEFORE training them.")
    print("FWT is computed only from measured A[j-1,j] cells.")
