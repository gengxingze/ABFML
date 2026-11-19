"""DataLoader factory combining AtomicDataset and neighbor settings into batches."""

from __future__ import annotations

from collections.abc import Mapping

from torch.utils.data import DataLoader, Dataset

from abfml.data.collate import StructureCollator
from abfml.core.neighbor import (
    NeighborBuilder,
    CachedNeighborBuilder,
    configure_cache,
    resolve_builder,
)
from abfml.core.structure import AtomicSample


def create_dataloader(
    dataset: Dataset[AtomicSample],
    provider: NeighborBuilder | None = None,
    *,
    model=None,
    neighbor_cache: str = "none",
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    label_scopes: Mapping[str, str] | None = None,
) -> DataLoader:
    """Create a PyTorch DataLoader yielding (ModelInput, label dictionary).
    
    Internal/none models skip external neighbors; external models select a backend by cutoff.
    Without a model, honor the explicit provider. neighbor_cache='memory' prewarms CPU graphs in
    the main process for reuse across batches and epochs; 'none' builds per batch. The dataset
    remains unchanged. StructureCollator supplies collate_fn; other arguments follow PyTorch
    DataLoader semantics. pin_memory currently pins labels only; ModelInput uses ordinary
    memory.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    provider = resolve_builder(model, provider)
    provider = configure_cache(provider, neighbor_cache)
    if isinstance(provider, CachedNeighborBuilder):
        provider.preload(dataset)
    collator = StructureCollator(
        provider,
        label_scopes=dict(label_scopes) if label_scopes is not None else None,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator,
    )
