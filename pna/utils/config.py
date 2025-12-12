"""Configuration management for PNA."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import yaml
import json


@dataclass
class MPEConfig:
    """MPE Core configuration."""
    state_dim: int = 64
    obs_dim: int = 64
    complexity_weight: float = 0.01
    learning_rate: float = 0.1
    precision_init: float = 1.0
    min_precision: float = 0.1
    max_precision: float = 10.0


@dataclass
class PolarityConfig:
    """Polarity Engine configuration."""
    input_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 32
    lambda_penalty: float = 1.0
    balance_threshold: float = 0.1
    octave_cycles: int = 8
    enable_rhythmic_modulation: bool = True


@dataclass
class BindingConfig:
    """Phenomenal Binding configuration."""
    num_nodes: int = 32
    node_dim: int = 32
    num_scales: int = 3
    consciousness_threshold: float = 0.5
    coupling_strength: float = 1.0
    history_length: int = 10


@dataclass
class PCConfig:
    """Predictive Coding configuration."""
    layer_dims: List[int] = field(default_factory=lambda: [784, 256, 128, 64])
    hidden_dim: int = 128
    activation: str = 'tanh'
    learnable_precision: bool = True
    inference_steps: int = 20
    learning_rate: float = 0.05


@dataclass
class TransparencyConfig:
    """Transparency configuration."""
    content_dim: int = 64
    mechanism_dim: int = 32
    repr_dim: int = 128
    state_dim: int = 64
    beta: float = 1.0
    hidden_dim: int = 128


@dataclass
class PNAConfig:
    """Complete PNA configuration."""
    # Architecture
    input_dim: int = 784
    mpe_dim: int = 64
    polarity_dim: int = 128
    binding_nodes: int = 32
    pc_layers: Optional[List[int]] = None

    # Thresholds
    consciousness_threshold: float = 0.5

    # Flags
    enable_transparency: bool = True

    # Loss weights
    polarity_lambda: float = 1.0
    binding_gamma: float = 1.0
    transparency_alpha: float = 0.5

    # Component configs
    mpe: MPEConfig = field(default_factory=MPEConfig)
    polarity: PolarityConfig = field(default_factory=PolarityConfig)
    binding: BindingConfig = field(default_factory=BindingConfig)
    pc: PCConfig = field(default_factory=PCConfig)
    transparency: TransparencyConfig = field(default_factory=TransparencyConfig)

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    device: str = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def save(self, filepath: str):
        """Save configuration to file."""
        ext = filepath.split('.')[-1]

        if ext == 'yaml' or ext == 'yml':
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        elif ext == 'json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {ext}")

    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary."""
        # Extract nested configs
        mpe = MPEConfig(**config_dict.get('mpe', {}))
        polarity = PolarityConfig(**config_dict.get('polarity', {}))
        binding = BindingConfig(**config_dict.get('binding', {}))
        pc = PCConfig(**config_dict.get('pc', {}))
        transparency = TransparencyConfig(**config_dict.get('transparency', {}))

        # Remove nested dicts
        main_config = {k: v for k, v in config_dict.items()
                      if k not in ['mpe', 'polarity', 'binding', 'pc', 'transparency']}

        return cls(
            **main_config,
            mpe=mpe,
            polarity=polarity,
            binding=binding,
            pc=pc,
            transparency=transparency
        )

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from file."""
        ext = filepath.split('.')[-1]

        if ext == 'yaml' or ext == 'yml':
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif ext == 'json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        return cls.from_dict(config_dict)


def load_config(filepath: str) -> PNAConfig:
    """Load PNA configuration from file."""
    return PNAConfig.load(filepath)


def get_default_config(dataset: str = 'mnist') -> PNAConfig:
    """Get default configuration for a dataset."""
    if dataset == 'mnist':
        return PNAConfig(
            input_dim=784,
            mpe_dim=64,
            polarity_dim=128,
            binding_nodes=32,
            pc_layers=[784, 256, 128, 64],
        )
    elif dataset == 'cifar10':
        return PNAConfig(
            input_dim=3072,  # 32x32x3
            mpe_dim=128,
            polarity_dim=256,
            binding_nodes=64,
            pc_layers=[3072, 1024, 512, 128],
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    # Example usage
    print("Testing configuration management...")

    # Create default config
    config = PNAConfig()
    print("\nDefault Config:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  MPE dim: {config.mpe_dim}")
    print(f"  Consciousness threshold: {config.consciousness_threshold}")
    print(f"  Device: {config.device}")

    # Save to file
    config.save('/tmp/pna_config.yaml')
    print("\nSaved to /tmp/pna_config.yaml")

    # Load from file
    loaded_config = PNAConfig.load('/tmp/pna_config.yaml')
    print(f"\nLoaded config matches: {config.to_dict() == loaded_config.to_dict()}")

    # Dataset-specific config
    mnist_config = get_default_config('mnist')
    print(f"\nMNIST config PC layers: {mnist_config.pc_layers}")

    cifar_config = get_default_config('cifar10')
    print(f"CIFAR-10 config PC layers: {cifar_config.pc_layers}")
