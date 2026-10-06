"""History and signal representation interfaces."""

from .abstraction import (
    MinimalSufficientRepresentation,
    QuotientRepresentation,
    StateAbstractionRepresentation,
    minimal_sufficient_representation,
)
from .advanced import (
    InformationBottleneckRepresentation,
    LearnedLatentRepresentation,
    PredictiveStateRepresentation,
    SharedMultimodalRepresentation,
    SupervisedBottleneckRepresentation,
    VariationalInformationBottleneck,
)
from .base import BaseRepresentation, RepresentationResult, RepresentationSpec
from .conditional import EmpiricalTaskLawRepresentation
from .controls import (
    RepresentationNegativeControl,
    permutation_control,
    randomized_representation_control,
)
from .information import RateDistortionRepresentation
from .standard import (
    FiniteHistoryRepresentation,
    HandcraftedPhysiologyRepresentation,
    InstantaneousStateRepresentation,
    MultiscaleHistoryRepresentation,
    PCARepresentation,
    PredictiveHistoryRepresentation,
    RandomProjectionRepresentation,
    RawSignalRepresentation,
    TaskSpecificRepresentation,
)
from .uncertainty import UncertaintyAwareRepresentation

__all__ = [
    "RepresentationSpec",
    "RepresentationResult",
    "BaseRepresentation",
    "RawSignalRepresentation",
    "InstantaneousStateRepresentation",
    "FiniteHistoryRepresentation",
    "PredictiveHistoryRepresentation",
    "MultiscaleHistoryRepresentation",
    "HandcraftedPhysiologyRepresentation",
    "PCARepresentation",
    "RandomProjectionRepresentation",
    "TaskSpecificRepresentation",
    "SupervisedBottleneckRepresentation",
    "InformationBottleneckRepresentation",
    "PredictiveStateRepresentation",
    "SharedMultimodalRepresentation",
    "VariationalInformationBottleneck",
    "LearnedLatentRepresentation",
    "RateDistortionRepresentation",
    "EmpiricalTaskLawRepresentation",
    "QuotientRepresentation",
    "StateAbstractionRepresentation",
    "MinimalSufficientRepresentation",
    "minimal_sufficient_representation",
    "UncertaintyAwareRepresentation",
    "RepresentationNegativeControl",
    "permutation_control",
    "randomized_representation_control",
]
