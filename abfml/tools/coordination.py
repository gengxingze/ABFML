"""Local neighbor counts and capacity analysis."""
from abfml.data import ASEReader
from abfml.core.neighbor import ASENeighborBuilder
from .common import plt, np, write_table, save_figure, summarize, output_dir, SCIENCE_STYLE, style_axes


def plot_neighbors(files, *, cutoffs, output=".", slots=None, export=False):
    """Plot one panel per cutoff; save only PNG by default and return summary statistics."""
    cutoffs = list(dict.fromkeys(cutoffs))
    if not cutoffs or any(not np.isfinite(c) or c <= 0 for c in cutoffs):
        raise ValueError('cutoffs must contain positive finite values')
    if slots is not None and slots < 1:
        raise ValueError('slots must be positive')
    out = output_dir(output)
    rows, stats, distributions = [], [], []
    for cutoff in cutoffs:
        builder = ASENeighborBuilder(cutoff)
        chunks = []
        for index, sample in enumerate(ASEReader(files)):
            graph = builder.build(sample)
            degrees = graph.degrees.numpy()
            chunks.append(degrees)
            if export:
                for center, degree in enumerate(degrees):
                    rows.append(dict(cutoff=cutoff, structure=index, center=center,
                                     center_element=int(sample.atomic_numbers[center]), neighbor_element="all",
                                     count=int(degree)))
                    start, end = graph.edge_ptr[center:center+2].tolist()
                    numbers = sample.atomic_numbers[graph.neighbor_indices[start:end]]
                    for z in sorted(set(sample.atomic_numbers.tolist())):
                        rows.append(dict(cutoff=cutoff, structure=index, center=center,
                                         center_element=int(sample.atomic_numbers[center]), neighbor_element=str(z),
                                         count=int((numbers==z).sum())))
        counts = np.concatenate(chunks) if chunks else np.array([])
        if not counts.size:
            raise ValueError("No atoms found")
        overflow = float(np.mean(counts > slots)) if slots is not None else None
        stats.append(dict(cutoff=cutoff, structures=len(chunks),
                          overflow_fraction=overflow if overflow is not None else "N/A",
                          overflow_percent=100.*overflow if overflow is not None else "N/A", **summarize(counts)))
        distributions.append(counts)
    ncols = min(3, len(cutoffs))
    nrows = (len(cutoffs)+ncols-1)//ncols
    with plt.rc_context(SCIENCE_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False, layout="constrained")
        try:
            for index, (cutoff, counts) in enumerate(zip(cutoffs, distributions)):
                ax = axes.flat[index]
                ax.hist(counts, bins=np.arange(counts.max()+2)-.5, color="#0072B2", alpha=.8)
                ax.set(xlabel="Neighbors per atom", ylabel="Count", title=f"({chr(97+index)}) Cutoff = {cutoff:g} Å")
                style_axes(ax)
            for ax in list(axes.flat)[len(cutoffs):]:
                fig.delaxes(ax)
            save_figure(fig, out / "neighbors.png", dpi=300)
        finally:
            plt.close(fig)
    if export:
        write_table(out / "neighbors.csv", rows)
        write_table(out / "neighbors_summary.csv", stats)
    return stats
