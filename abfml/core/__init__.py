"""Core interfaces without importing concrete models or ASE."""

from abfml.core.structure import AtomicSample
from abfml.core.model import AtomicModel
from abfml.core.types import ModelInput, ModelOutput
from abfml.core.schema import LabelSpec, LABELS
from abfml.core.neighbor import CSRNeighbors, DenseNeighbors, NeighborBuilder, to_dense
