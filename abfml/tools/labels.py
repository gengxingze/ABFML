"""Dataset label extraction and distribution comparison."""
from collections.abc import Mapping

from abfml.data import ASEReader
from .common import plt, np, write_table, save_figure, summarize, output_dir, SCIENCE_STYLE, style_axes
from abfml.core.schema import LABELS, label_unit

def read_labels(files):
    result = {}
    reader = ASEReader(files)
    count = 0
    def add(name, value):
        result.setdefault(name, []).append(np.asarray(value).ravel())
    for sample in reader:
        count += 1
        n = sample.num_atoms
        if n == 0:
            raise ValueError("Empty structures are unsupported")
        for name, spec in LABELS.items():
            if spec.source is not None:
                continue
            value = sample.get_label(name)
            if value is not None:
                add(name, value.numpy())
                if spec.per_atom:
                    add(spec.per_atom, value.numpy() / n)
        if sample.forces is not None:
            add("force_norm", sample.forces.norm(dim=1).numpy())
        if sample.virial is not None:
            volume = abs(np.linalg.det(sample.cell.numpy()))
            if volume > 0:
                add("stress", -sample.virial.numpy() / volume)
    paths = ", ".join(str(path) for path in reader.files)
    if not count:
        raise ValueError(f"No structures read from {paths}; check the XYZ atom-count header and file format")
    if not result:
        raise ValueError(f"No supported labels found in {paths}")
    return {name: np.concatenate(parts) for name, parts in result.items()}

def plot_distributions(groups, *, output=".", bins=60, export=False):
    """Plot extracted label groups using shared bin edges."""
    if bins < 1:
        raise ValueError('bins must be positive')
    if not groups or not any(groups.values()):
        raise ValueError('No label groups provided')
    out = output_dir(output)
    rows = []
    labels = sorted(set().union(*(g.keys() for g in groups.values())))
    ncols = min(3, len(labels))
    with plt.rc_context(SCIENCE_STYLE):
        fig, axes = plt.subplots((len(labels)+ncols-1)//ncols, ncols,
                                 figsize=(4*ncols, 3*((len(labels)+ncols-1)//ncols)),
                                 squeeze=False, layout="constrained")
        try:
            for index, name in enumerate(labels):
                ax = axes.flat[index]
                available = {group: values[name] for group, values in groups.items() if name in values}
                edges = np.histogram_bin_edges(np.concatenate(list(available.values())), bins=bins)
                hist_rows = []
                for group, values in available.items():
                    unit = label_unit(name)
                    rows.append(dict(dataset=group, label=name, unit=unit, **summarize(values)))
                    ax.hist(values, bins=edges, density=True, histtype="step", linewidth=1.5, label=group)
                    if export:
                        counts, _ = np.histogram(values, bins=edges)
                        hist_rows.extend(dict(dataset=group, left=edges[i], right=edges[i+1], count=int(c))
                                         for i, c in enumerate(counts))
                ax.set(xlabel=f"{name} ({label_unit(name).replace('A', 'Å')})",
                       ylabel="Probability density", title=f"({chr(97+index)}) {name.replace('_', ' ')}")
                style_axes(ax)
                ax.legend(frameon=False)
                if export:
                    write_table(out / f"{name}_histogram.csv", hist_rows)
            for ax in list(axes.flat)[len(labels):]:
                fig.delaxes(ax)
            save_figure(fig, out / "labels.png", dpi=300)
        finally:
            plt.close(fig)
    if export:
        write_table(out / "labels_summary.csv", rows)
    return rows


def plot_labels(files, *, output=".", bins=60, export=False):
    """Merge paths into one group, or preserve named path groups for comparison."""
    datasets = files if isinstance(files, Mapping) else {'dataset': files}
    groups = {}
    for name, paths in datasets.items():
        try:
            groups[name] = read_labels(paths)
        except ValueError as error:
            raise ValueError(f"Dataset '{name}': {error}") from error
    return plot_distributions(groups, output=output, bins=bins, export=export)
