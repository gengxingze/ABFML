"""PyTorch dataset storing structures and labels only."""

from __future__ import annotations

from collections.abc import Iterable
from torch.utils.data import Dataset

from abfml.core.structure import AtomicSample


class AtomicDataset(Dataset[AtomicSample]):
    """Immutable sequence wrapper around AtomicSample, without model-specific neighbors.
    
    Stores structures and supervision labels and provides indexing. Readers handle file parsing;
    arbitrary sample iterables are accepted. The sequence is immutable, but individual samples
    and tensors need not be.
    """

    def __init__(self, structures: Iterable[AtomicSample]) -> None:
        structures = tuple(structures)
        if any(not isinstance(item, AtomicSample) for item in structures):
            raise TypeError("All dataset items must be AtomicSample instances")
        self._structures = structures

    def __len__(self) -> int:
        return len(self._structures)

    def __getitem__(self, index: int) -> AtomicSample:
        return self._structures[index]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_structures={len(self)})"

    @property
    def structures(self) -> tuple[AtomicSample, ...]:
        """Return the internal sequence without copying structures or tensors."""
        return self._structures

    def subset(self, indices: Iterable[int]) -> AtomicDataset:
        """Create a subset from explicit indices without copying tensors.
        
        Negative indices, input order, and duplicates are preserved.
        """
        selected = []
        for index in indices:
            if not isinstance(index, int):
                raise TypeError("Dataset indices must be integers")
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError(f"Dataset index out of range: {index}")
            selected.append(self._structures[index])
        return type(self)(selected)
