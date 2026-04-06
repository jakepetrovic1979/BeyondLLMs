"""
Enhanced Level-2 Meta-Controller with Per-Parameter Plasticity
================================================================

CRITICAL ENHANCEMENT to address reviewer concern:
"scalar plasticity is likely too weak for your stated HOPE-style interpretation"

Solution: Replace scalar gate with:
1. Per-parameter plasticity gates (vector, not scalar)
2. Multi-timescale traces (fast/slow like Adam moments)
3. Memory retrieval analysis (t-SNE of meta-state by task)

This provides stronger evidence that L2 is an "optimizer as associative memory"
consistent with Google's Nested Learning / HOPE framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional, Tuple
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pna_complete_implementation_v2 import (
    MPECore, PolarityEngine, GlobalWorkspace,
    get_split_mnist_loaders
)


class EnhancedMetaController(nn.Module):
    """
    Level-2 Enhanced: Per-parameter plasticity gates with multi-timescale memory.

    Key differences from scalar version:
    - Outputs per-layer plasticity gates (not single scalar)
    - Maintains fast/slow traces (gradient history compression)
    - Memory retrieval: meta-state clusters by gradient regime

    Inspired by:
    - Google Nested Learning (optimizer as learned memory)
    - HOPE (Higher-Order Projection Estimation)
    - Adam moments (multi-timescale trace)
    """

    def __init__(
        self,
        d_model: int = 64,
        d_hidden: int = 128,
        num_parameter_groups: int = 4,  # L1 has ~4 parameter groups
        update_freq: int = 5,
        trace_decay_fast: float = 0.9,  # Like β1 in Adam
        trace_decay_slow: float = 0.999,  # Like β2 in Adam
        device: str = 'cpu'
    ):
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_groups = num_parameter_groups
        self.update_freq = update_freq
        self.device = device

        # Multi-timescale trace decay
        self.beta_fast = trace_decay_fast
        self.beta_slow = trace_decay_slow

        # Memory: GRU accumulates gradient statistics
        self.meta_gru = nn.GRU(
            input_size=num_parameter_groups * 2,  # [grad_norm, loss] per group
            hidden_size=d_hidden,
            batch_first=True
        ).to(device)

        # Gate generator: per-parameter-group plasticity
        self.gate_projector = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, num_parameter_groups),
            nn.Sigmoid()  # Gates in [0, 1]
        ).to(device)

        # Multi-timescale traces (gradient history)
        self.register_buffer('trace_fast', torch.zeros(num_parameter_groups, device=device))
        self.register_buffer('trace_slow', torch.zeros(num_parameter_groups, device=device))

        # Meta-state history for memory analysis
        self.meta_states = []  # Store for t-SNE analysis
        self.meta_labels = []  # Task/regime labels

        # Internal state
        self.h_meta = torch.zeros(1, 1, d_hidden, device=device)
        self.step_count = 0

    def forward(
        self,
        parameter_groups: list,  # List of parameter tensors
        loss: torch.Tensor,
        task_id: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-group plasticity gates.

        Args:
            parameter_groups: List of parameter groups from L1
            loss: Current loss value
            task_id: Optional task label for memory analysis

        Returns:
            dict with 'gates' (per-group), 'traces' (fast/slow), 'meta_state'
        """
        self.step_count += 1

        # Extract gradient statistics per group
        grad_norms = []
        for group in parameter_groups:
            if group.grad is not None:
                norm = group.grad.norm().item()
            else:
                norm = 0.0
            grad_norms.append(norm)

        # Pad if needed
        while len(grad_norms) < self.num_groups:
            grad_norms.append(0.0)
        grad_norms = grad_norms[:self.num_groups]

        # Update multi-timescale traces
        grad_tensor = torch.tensor(grad_norms, device=self.device)
        self.trace_fast = self.beta_fast * self.trace_fast + (1 - self.beta_fast) * grad_tensor
        self.trace_slow = self.beta_slow * self.trace_slow + (1 - self.beta_slow) * grad_tensor**2

        # Create input: [grad_norm, loss] per group
        loss_vec = loss.item() * torch.ones(self.num_groups, device=self.device)
        meta_input = torch.stack([grad_tensor, loss_vec], dim=1)  # [num_groups, 2]
        meta_input = meta_input.unsqueeze(0)  # [1, num_groups, 2]

        # GRU update
        _, self.h_meta = self.meta_gru(meta_input, self.h_meta)

        # Generate per-group plasticity gates
        gates = self.gate_projector(self.h_meta.squeeze(0))  # [num_groups]

        # Store meta-state for memory analysis
        if task_id is not None:
            self.meta_states.append(self.h_meta.squeeze().detach().cpu().numpy())
            self.meta_labels.append(task_id)

        return {
            'gates': gates.squeeze(),  # [num_groups]
            'trace_fast': self.trace_fast.clone(),
            'trace_slow': self.trace_slow.clone(),
            'meta_state': self.h_meta.clone()
        }

    def reset(self):
        """Reset hidden state (between episodes)."""
        self.h_meta = torch.zeros_like(self.h_meta)
        self.step_count = 0

    def get_memory_analysis(self) -> Dict[str, np.ndarray]:
        """
        Extract meta-state history for memory retrieval analysis.

        Returns:
            dict with 'states' [N, d_hidden] and 'labels' [N]
        """
        if len(self.meta_states) == 0:
            return {'states': np.array([]), 'labels': np.array([])}

        states = np.array(self.meta_states)
        labels = np.array(self.meta_labels)

        return {'states': states, 'labels': labels}

    def clear_memory_history(self):
        """Clear stored meta-states (for fresh analysis)."""
        self.meta_states = []
        self.meta_labels = []


class PNASystemWithEnhancedL2(nn.Module):
    """
    PNA with enhanced Level-2 meta-controller.

    Key changes:
    - L2 outputs per-layer gates (not scalar)
    - Apply gates selectively to L1 parameter groups
    - Track multi-timescale traces
    """

    def __init__(
        self,
        d_input: int = 28*28,
        d_hidden: int = 64,
        d_output: int = 2,
        lambda_pol: float = 0.1,
        lambda_mpe: float = 0.5,
        device: str = 'cpu'
    ):
        super().__init__()

        self.device = device

        # Level 0: MPE Core
        self.l0_mpe = MPECore(d_input, d_hidden, device=device)

        # Level 1: Polarity Engine
        self.l1_polarity = PolarityEngine(d_hidden, device=device)

        # Level 2: ENHANCED Meta-Controller
        param_groups = [
            self.l1_polarity.gru_g.weight_hh_l0,
            self.l1_polarity.gru_r.weight_hh_l0,
            self.l1_polarity.gate_network[0].weight,
            self.l1_polarity.gate_network[2].weight
        ]
        self.l2_meta = EnhancedMetaController(
            d_model=d_hidden,
            num_parameter_groups=len(param_groups),
            device=device
        )

        # Level 3: Global Workspace
        self.l3_workspace = GlobalWorkspace(d_hidden, device=device)

        # Output head
        self.output_head = nn.Linear(d_hidden, d_output).to(device)

        # Hyperparameters
        self.lambda_pol = lambda_pol
        self.lambda_mpe = lambda_mpe

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        task_id: Optional[int] = None
    ) -> Dict:
        """Forward with enhanced L2."""

        # Reshape input
        x_seq = x.unsqueeze(1)  # [batch_size, 1, d_input]

        # L0: MPE
        z_mpe = self.l0_mpe(x_seq)

        # L1: Polarity
        polarity_result = self.l1_polarity(z_mpe)
        z_pol = polarity_result['z']
        h_pol = polarity_result['h']

        # L3: Workspace
        workspace_result = self.l3_workspace(z_pol, h_pol)
        w = workspace_result['w']

        # Output
        output = self.output_head(w.squeeze(1))

        # Loss computation
        if targets is not None:
            task_loss = F.cross_entropy(output, targets)
            polarity_loss = polarity_result['polarity_loss']
            mpe_loss = self.l0_mpe.f0_current

            total_loss = (
                task_loss +
                self.lambda_pol * polarity_loss +
                self.lambda_mpe * mpe_loss
            )

            # L2: Compute per-group plasticity gates
            param_groups = [
                self.l1_polarity.gru_g.weight_hh_l0,
                self.l1_polarity.gru_r.weight_hh_l0,
                self.l1_polarity.gate_network[0].weight,
                self.l1_polarity.gate_network[2].weight
            ]

            meta_result = self.l2_meta(param_groups, total_loss, task_id)
            gates = meta_result['gates']

        else:
            total_loss = None
            gates = None
            meta_result = None

        return {
            'output': output.unsqueeze(1),  # For consistency
            'workspace': w,
            'polarity_state': h_pol,
            'total_loss': total_loss,
            'plasticity_gates': gates,  # Per-group
            'meta_traces': meta_result['trace_fast'] if meta_result else None,
            'phi': workspace_result.get('phi', 0.0)
        }

    def apply_plasticity_gates(self, gates: torch.Tensor):
        """
        Apply per-group plasticity gates to L1 gradients.

        CRITICAL: This is per-parameter, not scalar.
        """
        param_groups = [
            self.l1_polarity.gru_g.weight_hh_l0,
            self.l1_polarity.gru_r.weight_hh_l0,
            self.l1_polarity.gate_network[0].weight,
            self.l1_polarity.gate_network[2].weight
        ]

        for i, param in enumerate(param_groups):
            if i < len(gates) and param.grad is not None:
                gate_value = gates[i].item()
                param.grad *= gate_value

    def reset(self):
        """Reset all internal states."""
        self.l0_mpe.reset()
        self.l1_polarity.reset()
        self.l2_meta.reset()
        self.l3_workspace.reset()


# ============================================================================
# TRAINING EXAMPLE with Enhanced L2
# ============================================================================

def train_with_enhanced_l2(num_tasks: int = 5, epochs_per_task: int = 3):
    """
    Training loop demonstrating:
    1. Per-parameter plasticity gates
    2. Multi-timescale traces
    3. Memory state collection for t-SNE analysis
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PNASystemWithEnhancedL2(d_input=28*28, d_hidden=64, d_output=2, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training with Enhanced L2 (per-parameter gates + multi-timescale traces)")
    print("="*80)

    for task_id in range(num_tasks):
        print(f"\nTask {task_id + 1}/{num_tasks}")
        train_loader, test_loader = get_split_mnist_loaders(task_id, batch_size=32, device=device)

        for epoch in range(epochs_per_task):
            model.train()

            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)

                # Forward
                result = model(data, target, task_id=task_id)

                # Backward
                optimizer.zero_grad()
                result['total_loss'].backward()

                # Apply per-parameter gates
                if result['plasticity_gates'] is not None:
                    model.apply_plasticity_gates(result['plasticity_gates'])

                # Update
                optimizer.step()
                model.reset()

                if batch_idx % 50 == 0:
                    gates = result['plasticity_gates'].cpu().numpy()
                    traces = result['meta_traces'].cpu().numpy()

                    print(f"  Batch {batch_idx}: Loss={result['total_loss'].item():.4f}, "
                          f"Gates={gates}, Traces_fast={traces[:2]}")

    # Memory analysis
    print("\n" + "="*80)
    print("MEMORY RETRIEVAL ANALYSIS")
    print("="*80)

    memory_data = model.l2_meta.get_memory_analysis()
    states = memory_data['states']
    labels = memory_data['labels']

    print(f"Collected {len(states)} meta-states across {num_tasks} tasks")
    print(f"Meta-state shape: {states[0].shape if len(states) > 0 else 'N/A'}")

    # Save for t-SNE visualization
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'l2_meta_states.npy'), states)
    np.save(os.path.join(output_dir, 'l2_meta_labels.npy'), labels)

    print(f"\nSaved meta-states to {output_dir}/l2_meta_states.npy")
    print("Use t-SNE to visualize clustering by task (evidence of memory retrieval)")

    return model, states, labels


if __name__ == "__main__":
    model, states, labels = train_with_enhanced_l2(num_tasks=5, epochs_per_task=2)

    print("\n" + "="*80)
    print("ABLATION: Compare scalar vs per-parameter gates")
    print("="*80)
    print("Run experiments with:")
    print("1. Scalar gate (original)")
    print("2. Per-parameter gates (this version)")
    print("3. No gates (baseline)")
    print("\nExpected: Per-parameter gates provide finer control → better retention")
