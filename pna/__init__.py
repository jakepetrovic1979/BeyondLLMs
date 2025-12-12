"""Phenomenal Nested Architectures (PNA) - Consciousness-First AGI Framework."""

__version__ = "0.1.0"

from pna.core.mpe_core import MPECore
from pna.core.polarity_engine import PolarityEngine
from pna.core.binding import PhenomenalBindingMechanism
from pna.core.predictive_coding import PredictiveCodingHierarchy
from pna.core.transparency import TransparencyBottleneck
from pna.models.pna_model import PhenomenalNestedArchitecture

__all__ = [
    "MPECore",
    "PolarityEngine",
    "PhenomenalBindingMechanism",
    "PredictiveCodingHierarchy",
    "TransparencyBottleneck",
    "PhenomenalNestedArchitecture",
]
