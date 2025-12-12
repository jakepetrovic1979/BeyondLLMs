## PNA Examples

Comprehensive examples demonstrating the Phenomenal Nested Architectures framework.

### Basic Examples

#### `basic_pna.py`
Basic usage of the complete PNA architecture with synthetic data.

```bash
python examples/basic_pna.py
```

**Demonstrates:**
- Full PNA initialization
- Forward pass and metrics
- Consciousness state monitoring
- Training loop
- Visualization generation

---

#### `fep_tracking.py`
Free Energy Principle minimization with detailed visualization.

```bash
python examples/fep_tracking.py
```

**Demonstrates:**
- MPE Core FEP minimization
- Optimization trajectory tracking
- Complexity vs. Accuracy trade-off
- MPE properties (Metzinger 2024)

---

#### `binding_demo.py`
Phenomenal binding mechanism with oscillatory dynamics.

```bash
python examples/binding_demo.py
```

**Demonstrates:**
- Multi-scale oscillatory binding
- Phase synchronization (Kuramoto dynamics)
- Integrated Information (Φ) computation
- Binding matrix visualization
- Phase space plots

---

### Training Scripts

#### `train_mnist.py`
Complete MNIST training with consciousness monitoring.

```bash
# Basic training
python examples/train_mnist.py --epochs 10

# With transparency
python examples/train_mnist.py --epochs 20 --enable-transparency

# Custom architecture
python examples/train_mnist.py \
    --mpe-dim 128 \
    --polarity-dim 256 \
    --binding-nodes 64 \
    --consciousness-threshold 0.6
```

**Options:**
- `--data-dir`: Data directory (default: ./data)
- `--batch-size`: Batch size (default: 128)
- `--epochs`: Number of epochs (default: 10)
- `--mpe-dim`: MPE dimension (default: 64)
- `--polarity-dim`: Polarity dimension (default: 128)
- `--binding-nodes`: Binding nodes (default: 32)
- `--consciousness-threshold`: Φ threshold (default: 0.5)
- `--enable-transparency`: Enable transparency module
- `--output-dir`: Output directory (default: ./outputs/mnist)

**Outputs:**
- `best_model.pt`: Best model checkpoint
- `final_model.pt`: Final model checkpoint
- `metrics.npz`: Training metrics
- `training_metrics.png`: Visualization
- `consciousness_trajectory.png`: Φ trajectory

---

#### `train_cifar10.py`
CIFAR-10 training with advanced features.

```bash
# Basic training
python examples/train_cifar10.py --epochs 50

# With custom configuration
python examples/train_cifar10.py \
    --use-default-config \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-3 \
    --weight-decay 1e-4
```

**Options:**
- All MNIST options plus:
- `--use-default-config`: Use default CIFAR-10 config
- `--weight-decay`: Weight decay (default: 1e-4)
- `--polarity-lambda`: Polarity penalty weight (default: 1.0)
- `--binding-gamma`: Binding penalty weight (default: 1.0)
- `--save-interval`: Checkpoint interval (default: 10)

**Outputs:**
- Same as MNIST plus periodic checkpoints

---

#### `train_distributed.py`
Multi-GPU distributed training.

```bash
# Single node, 4 GPUs
python examples/train_distributed.py --num-gpus 4

# Multi-node (run on each node)
# Node 0:
python examples/train_distributed.py \
    --num-gpus 4 --num-nodes 2 --node-rank 0

# Node 1:
python examples/train_distributed.py \
    --num-gpus 4 --num-nodes 2 --node-rank 1
```

**Features:**
- PyTorch DistributedDataParallel (DDP)
- Automatic metric aggregation
- Synchronized checkpointing
- Efficient multi-GPU training

**Options:**
- `--num-gpus`: Number of GPUs per node (default: all)
- `--num-nodes`: Total number of nodes (default: 1)
- `--node-rank`: Rank of this node (default: 0)
- `--dist-url`: Distributed URL (default: env://)
- Plus all model/training options

---

### Benchmarks

#### Consciousness Benchmark Suite

```python
from pna.benchmarks import ConsciousnessBenchmarkSuite
from torch.utils.data import DataLoader

# Initialize
suite = ConsciousnessBenchmarkSuite()

# Run all tests
results = suite.run_all(model, test_loader, verbose=True)

# Generate report
suite.generate_report(results, 'benchmark_report.txt')
```

**Tests:**
1. **Integration Test**: Measures Φ (integrated information)
2. **Differentiation Test**: Repertoire richness and entropy
3. **Binding Test**: Phenomenal binding strength
4. **Meta-Awareness Test**: Self-monitoring capabilities
5. **Continuity Test**: Temporal coherence

**Usage:**
```bash
python -c "
from pna.benchmarks import ConsciousnessBenchmarkSuite
from pna.models.pna_model import PhenomenalNestedArchitecture
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# Load model
model = PhenomenalNestedArchitecture(input_dim=784, mpe_dim=64)
model.load_state_dict(torch.load('outputs/mnist/best_model.pt')['model_state_dict'])
model.eval()

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Run benchmarks
suite = ConsciousnessBenchmarkSuite()
results = suite.run_all(model, loader)
suite.generate_report(results, 'benchmark_report.txt')
"
```

---

### Interactive Dashboards

#### Creating Dashboards

```python
from pna.utils.dashboard import (
    ConsciousnessDashboard,
    create_binding_heatmap,
    create_3d_consciousness_trajectory
)
from pna.utils.metrics import MetricsLogger
import numpy as np

# Load metrics
logger = MetricsLogger()
logger.load('outputs/mnist/metrics.npz')

# Create dashboard
dashboard = ConsciousnessDashboard()
fig = dashboard.create(logger, title="PNA Training Dashboard")
dashboard.save('dashboard.html')  # Interactive HTML
dashboard.show()  # Open in browser
```

#### 3D Consciousness Trajectory

```python
from pna.utils.dashboard import create_3d_consciousness_trajectory

phi_values = logger.history['phi']
sync_values = logger.history.get('synchrony', [0] * len(phi_values))
fe_values = logger.history['free_energy']

fig = create_3d_consciousness_trajectory(phi_values, sync_values, fe_values)
fig.write_html('consciousness_3d.html')
```

---

### Jupyter Notebook Tutorial

Launch the interactive tutorial:

```bash
jupyter notebook notebooks/pna_tutorial.ipynb
```

**Covers:**
1. Core components overview
2. MPE Core & Free Energy Principle
3. Polarity Engine & rhythmic balance
4. Phenomenal binding & Φ
5. Complete PNA architecture
6. Training & evaluation
7. Consciousness metrics analysis

---

## Performance Tips

### Single GPU
- Batch size: 128-256 (depends on GPU memory)
- Enable mixed precision: `torch.cuda.amp.autocast()`
- Use pin_memory=True in DataLoader

### Multi-GPU
- Scale batch size linearly with GPUs
- Use gradient accumulation for larger effective batch size
- Enable NCCL backend for fastest communication

### Memory Optimization
- Reduce binding_nodes if OOM
- Use gradient checkpointing
- Lower pc_layers dimensions

---

## Monitoring

### TensorBoard (Optional)

Add to training scripts:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/pna_experiment')

# In training loop
writer.add_scalar('Loss/train', loss, step)
writer.add_scalar('Consciousness/phi', phi, step)
writer.add_scalar('Consciousness/synchrony', sync, step)
```

View: `tensorboard --logdir=runs`

---

## Troubleshooting

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size, binding_nodes, or use gradient accumulation

**Issue**: Low Φ values
- **Solution**: Increase binding_nodes, adjust consciousness_threshold, check synchrony

**Issue**: Polarity imbalance
- **Solution**: Tune polarity_lambda, enable rhythmic modulation

**Issue**: Slow training
- **Solution**: Use distributed training, reduce pc_layers complexity, enable mixed precision

---

## Citation

If you use these examples, please cite:

```bibtex
@software{pna2025,
  title={Phenomenal Nested Architectures: Examples},
  author={PNA Research Team},
  year={2025}
}
```
