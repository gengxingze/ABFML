"""Descriptor channel statistics and shared PCA."""
import torch
from abfml.data import ASEReader, AtomicDataset, create_dataloader
from abfml.serialization import load_model
from .common import plt, np, write_table, save_figure, summarize, output_dir, SCIENCE_STYLE, style_axes

def plot_descriptors(model_path, datasets, *, output=".", batch_size=8, export=False, level="atom"):
    """Analyze atomic descriptors or structure means, fitting one shared PCA across groups."""
    if level not in {"atom", "structure"}:
        raise ValueError('level must be atom or structure')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if not datasets:
        raise ValueError('No datasets provided')
    out = output_dir(output)
    model = load_model(model_path)
    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.float64
    chunks, metadata = [], []
    group_counts = {}
    for group, files in datasets.items():
        dataset = AtomicDataset(ASEReader(files, dtype=dtype))
        group_counts[group] = dict(structures=len(dataset), centers=0)
        offset = 0
        for inputs, _ in create_dataloader(dataset, model=model, batch_size=batch_size):
            with torch.no_grad():
                output = model.compute(inputs)
            if "descriptors" not in output.properties:
                raise ValueError("Model does not expose properties['descriptors']")
            values = output.properties["descriptors"].detach().cpu().numpy()
            if values.ndim != 2 or values.shape[0] != inputs.center_indices.numel():
                raise ValueError("Descriptors must have shape [centers, channels]")
            group_counts[group]["centers"] += values.shape[0]
            batches = inputs.center_batch.cpu().numpy()
            if level == "structure":
                counts = np.bincount(batches, minlength=inputs.num_structures)
                if (counts == 0).any():
                    raise ValueError("Structure descriptors require at least one center per structure")
                pooled = np.zeros((inputs.num_structures, values.shape[1]), dtype=np.float64)
                np.add.at(pooled, batches, values)
                chunks.append(pooled/counts[:, None])
                for index, count in enumerate(counts):
                    metadata.append(dict(dataset=group, structure=offset+index, level=level, centers=int(count)))
            else:
                chunks.append(values)
                for row, center in enumerate(inputs.center_indices.tolist()):
                    batch = int(batches[row])
                    metadata.append(dict(dataset=group, structure=offset+batch, level=level,
                                         atom=center-int(inputs.atom_ptr[batch]),
                                         element=int(inputs.atomic_numbers[center])))
            offset += inputs.num_structures
    if not chunks:
        raise ValueError('No descriptor rows found')
    x = np.concatenate(chunks)
    if x.shape[0]<2 or not np.isfinite(x).all():
        raise ValueError(f"PCA requires at least two finite {level}-level descriptor rows")
    # Shared centered PCA, without feature standardization, across every dataset.
    centered = x.astype(float)-x.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[:2].T
    if scores.shape[1]<2:
        scores = np.pad(scores, ((0,0),(0,1)))
    variance = singular**2
    ratios = variance/variance.sum() if variance.sum() else np.zeros_like(variance)
    if export:
        np.savez_compressed(out / "descriptors.npz", descriptors=x, mean=x.mean(axis=0),
                            components=vt[:2], singular_values=singular, level=level,
                            pooling="mean" if level == "structure" else "none")
        write_table(out / "pca.csv", [dict(**m, PC1=s[0], PC2=s[1]) for m,s in zip(metadata,scores)])
        write_table(out / "explained_variance.csv", [dict(component=i+1, ratio=v) for i,v in enumerate(ratios)])
        write_table(out / "channels.csv", [dict(channel=i, **summarize(x[:,i])) for i in range(x.shape[1])])
    with plt.rc_context(SCIENCE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), layout="constrained")
        try:
            for group in datasets:
                mask = np.array([m["dataset"]==group for m in metadata])
                axes[0].scatter(scores[mask,0], scores[mask,1], s=5, alpha=.4, label=group)
            axes[0].set(xlabel=f"PC1 ({ratios[0]*100:.2f}%)",
                        ylabel=f"PC2 ({ratios[1]*100 if len(ratios)>1 else 0:.2f}%)",
                        title=f"(a) {'Atomic environments' if level == 'atom' else 'Structures (mean pooling)'}")
            axes[0].legend(frameon=False)
            axes[1].plot(x.mean(axis=0), color="#0072B2", label="Mean")
            axes[1].fill_between(np.arange(x.shape[1]), x.mean(axis=0)-x.std(axis=0),
                                 x.mean(axis=0)+x.std(axis=0), color="#0072B2", alpha=.2, label="±1 std")
            axes[1].set(xlabel="Descriptor channel", ylabel="Value", title="(b) Descriptor channels")
            axes[1].legend(frameon=False)
            for ax in axes:
                style_axes(ax)
            save_figure(fig, out / "descriptors.png", dpi=300)
        finally:
            plt.close(fig)
    return [dict(dataset=group, level=level, points=sum(m['dataset']==group for m in metadata),
                 **group_counts[group], channels=x.shape[1],
                 pc1_percent=float(ratios[0]*100), pc2_percent=float(ratios[1]*100) if len(ratios)>1 else 0.)
            for group in datasets]
