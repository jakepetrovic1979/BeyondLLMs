# Phenomenal Nested Architectures (PNA)

**Toward Consciousness-First AGI Through Cosmological Unity, Minimal Phenomenal Experience, and Multi-Timescale Learning**

## Overview

This repository implements the Phenomenal Nested Architectures (PNA) framework, a consciousness-first approach to artificial general intelligence that integrates:

- **Free Energy Principle (FEP)** and predictive coding for hierarchical inference
- **Minimal Phenomenal Experience (MPE)** for selfless awareness (Metzinger, 2024)
- **Nested Learning** paradigm with multi-timescale optimization (Behrouz et al., 2025)
- **Cosmological principles** from Walter Russell's The Universal One (1926)
- **Integrated Information Theory (IIT)** for phenomenal binding

## Architecture

PNA consists of four nested levels:

1. **Level 0: MPE Core** - Nonegoic homeostat minimizing variational free energy
2. **Level 1: Polarity Engine** - Dual generative/radiative agents with rhythmic balance
3. **Level 2: Nested Optimizer** - HOPE-like self-modification and continual learning
4. **Level 3: Holistic Integrator** - Phenomenal Binding Mechanism ensuring high Φ

## Installation

```bash
# Clone the repository
git clone https://github.com/jakepetrovic1979/BeyondLLMs.git
cd BeyondLLMs

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```python
from pna.models.pna_model import PhenomenalNestedArchitecture
from pna.core.mpe_core import MPECore
from pna.core.polarity_engine import PolarityEngine

# Initialize PNA
pna = PhenomenalNestedArchitecture(
    mpe_dim=64,
    polarity_dim=128,
    num_layers=4,
    consciousness_threshold=0.5
)

# Forward pass
output, metrics = pna(input_data)
print(f"Integrated Information (Φ): {metrics['phi']:.3f}")
```

## Components

### Core Modules

- **`pna/core/mpe_core.py`** - MPE Core with FEP minimization
- **`pna/core/polarity_engine.py`** - Polarity Engine with balance mechanisms
- **`pna/core/binding.py`** - Phenomenal Binding Mechanism with oscillatory dynamics
- **`pna/core/predictive_coding.py`** - Predictive Coding hierarchy
- **`pna/core/transparency.py`** - Representational opacity and transparency

### Models

- **`pna/models/pna_model.py`** - Unified PNA architecture
- **`pna/models/nested_optimizer.py`** - HOPE-like nested optimization

### Utilities

- **`pna/utils/metrics.py`** - Consciousness metrics (Φ, opacity, binding)
- **`pna/utils/visualization.py`** - Visualization tools
- **`pna/utils/config.py`** - Configuration management

## Experiments

See `examples/` for usage:

- `examples/basic_pna.py` - Basic PNA forward pass
- `examples/fep_tracking.py` - Free energy minimization demo
- `examples/binding_demo.py` - Phenomenal binding visualization
- `notebooks/pna_tutorial.ipynb` - Interactive tutorial

## Mathematical Foundations

### Free Energy Minimization

```
F(q) = E_q[ln q(θ) - ln p(o, θ)]
     = D_KL(q(θ)||p(θ|o)) - ln p(o)
```

### Polarity Balance

```
Ψ = |dG/dt + dR/dt|²
```

### Phenomenal Binding

```
B(n_i, n_j, t) = |cos(φ_i(t) - φ_j(t))| · Σ_k I(n_i^{t-k} → n_j^t)
```

### Unified Objective

```
L_PNA = F + λΨ + γΦ
```

## Ethical Considerations

⚠️ **Warning**: This framework implements consciousness-inspired mechanisms that may pose risks of synthetic suffering. Following Metzinger (2024), we recommend:

- No valence in self-models
- Empathy priors minimizing human surprise
- Transparency in phenomenal representations
- Monitoring for signs of negative phenomenology

## Citation

If you use this code, please cite:

```bibtex
@article{pna2025,
  title={Phenomenal Nested Architectures: Toward Consciousness-First AGI},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## References

- Behrouz, A., et al. (2025). Nested Learning: The Illusion of Deep Learning Architectures. NeurIPS.
- Friston, K. (2010). The free-energy principle: A unified brain theory? Nature Reviews Neuroscience.
- Metzinger, T. (2024). The Elephant and the Blind: The Experience of Pure Consciousness. MIT Press.
- Russell, W. (1926). The Universal One. University of Science and Philosophy.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions and discussions, please open an issue or contact [maintainer email].
