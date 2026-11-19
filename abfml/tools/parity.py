"""Prediction panels and optional two-column numerical exports."""

import json
from pathlib import Path

from ase.io import iread

from .common import plt, np, SCIENCE_STYLE, style_axes, save_figure, output_dir
from abfml.core.schema import LABELS, label_unit


def collect_pairs(groups, references, predictions, num_atoms):
    """Collect reference/prediction pairs by label, including per-atom energy and virial."""
    if num_atoms <= 0:
        raise ValueError("num_atoms must be positive")
    for name in references.keys() & predictions.keys():
        ref = np.asarray(references[name], dtype=float)
        pred = np.asarray(predictions[name], dtype=float)
        if ref.shape != pred.shape or not np.isfinite(ref).all() or not np.isfinite(pred).all():
            raise ValueError(f"Invalid paired values: {name}")
        groups.setdefault(name, []).append((ref.ravel(), pred.ravel()))
        spec = LABELS.get(name)
        if spec is not None and spec.per_atom:
            groups.setdefault(spec.per_atom, []).append((ref.ravel()/num_atoms, pred.ravel()/num_atoms))


def read_pairs(path):
    """Read evaluation extxyz files or legacy JSONL predictions for replotting."""
    groups = {}
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                collect_pairs(groups, record["references"], record["predictions"], record["num_atoms"])
    else:
        for atoms in iread(path, format="extxyz"):
            references, predictions = {}, {}
            for key, shape in atoms.info.get("abfml_shapes", {}).items():
                value = atoms.arrays.get(key) if key in atoms.arrays else atoms.info.get(key)
                if value is None:
                    raise ValueError(f"Missing extxyz field: {key}")
                if key.startswith("ref_"):
                    references[key[4:]] = np.asarray(value).reshape(shape)
                elif key.startswith("pred_"):
                    predictions[key[5:]] = np.asarray(value).reshape(shape)
            collect_pairs(groups, references, predictions, len(atoms))
    return groups


def export_pairs(path, *, label, output):
    """Export reference and prediction columns, flattened in structure and component order."""
    pairs = read_pairs(path).get(label)
    if not pairs:
        raise ValueError(f"No paired values for label: {label}")
    destination = Path(output)
    if destination.resolve() == Path(path).resolve():
        raise ValueError("Output must differ from the prediction file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        handle.write("reference,prediction\n")
        for ref, pred in pairs:
            np.savetxt(handle, np.column_stack((ref, pred)), delimiter=",", fmt="%.16e")


def plot_predictions(source, *, output=".", labels=None):
    """Plot shared axes per label from extxyz files or named in-memory paired arrays."""
    datasets = {"Evaluation": read_pairs(source)} if isinstance(source, (str, Path)) else source
    available = set().union(*(group.keys() for group in datasets.values())) if datasets else set()
    if labels is None:
        selected = [name for name in available
                    if not (name in LABELS and LABELS[name].per_atom in available)]
        priority = {name: i for i, name in enumerate(("energy_per_atom", "forces", "virial_per_atom", "charges"))}
        selected.sort(key=lambda name: (priority.get(name, 99), name))
    else:
        selected = list(labels)
        if not selected or set(selected) - available:
            raise ValueError("Requested plot labels must have paired reference and prediction values")
    if not selected:
        return []
    titles = {"energy": "Total energy", "energy_per_atom": "Energy per atom",
              "forces": "Force components", "virial": "Virial components",
              "virial_per_atom": "Virial per atom", "charges": "Atomic charges"}
    out = output_dir(output)
    ncols = min(3, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    statistics = []
    with plt.rc_context(SCIENCE_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.1*ncols, 4.0*nrows),
                                 squeeze=False, layout="constrained")
        try:
            for panel, name in enumerate(selected):
                ax = axes.flat[panel]
                series = {}
                for group, values in datasets.items():
                    if values.get(name):
                        ref = np.concatenate([pair[0] for pair in values[name]])
                        pred = np.concatenate([pair[1] for pair in values[name]])
                        if ref.shape != pred.shape or not ref.size or not np.isfinite([ref, pred]).all():
                            raise ValueError(f"Invalid paired values: {name}")
                        series[group] = (ref, pred)
                lo = min(min(ref.min(), pred.min()) for ref, pred in series.values())
                hi = max(max(ref.max(), pred.max()) for ref, pred in series.values())
                padding = .06*(hi-lo) if hi > lo else .06*max(abs(lo), 1.)
                limits = (lo-padding, hi+padding)
                unit = label_unit(name).replace("A", "Å")
                for index, (group, (ref, pred)) in enumerate(series.items()):
                    color = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")[index % 4]
                    if len(series) == 1 and ref.size > 1000:
                        density = ax.hexbin(ref, pred, gridsize=60, mincnt=1, bins="log", cmap="viridis", linewidths=0)
                        fig.colorbar(density, ax=ax, fraction=.046, pad=.04).set_label("Samples per bin")
                    else:
                        ax.scatter(ref, pred, s=9, alpha=.4, color=color, edgecolors="none", label=group)
                    residual = pred-ref
                    statistics.append(dict(dataset=group, label=name, unit=unit, count=ref.size,
                                           rmse=float(np.sqrt(np.mean(residual**2))), mae=float(np.abs(residual).mean())))
                    text = (f"{group}: N = {ref.size:,}\n"
                            f"RMSE = {np.sqrt(np.mean(residual**2)):.3g} {unit}\n"
                            f"MAE = {np.abs(residual).mean():.3g} {unit}")
                    ax.text(.04, .96-index*.25, text, transform=ax.transAxes, va="top", fontsize=8,
                            color=color if len(series) > 1 else "black",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=2))
                ax.plot(limits, limits, "--", color="0.25", linewidth=1.)
                ax.set(xlim=limits, ylim=limits, xlabel=f"Reference ({unit})", ylabel=f"Prediction ({unit})",
                       title=f"({chr(97+panel)})  {titles.get(name, name.replace('_', ' ').capitalize())}")
                ax.set_aspect("equal", adjustable="box")
                ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4), useOffset=False)
                style_axes(ax)
            # Use one figure-level legend so group labels do not obscure panel data.
            if len(datasets) > 1:
                handles, names = [], []
                for axis in axes.flat:
                    axis_handles, axis_names = axis.get_legend_handles_labels()
                    for handle, name in zip(axis_handles, axis_names):
                        if name not in names:
                            handles.append(handle)
                            names.append(name)
                fig.legend(handles, names, loc="outside lower center", ncol=len(names), frameon=False)
            for ax in list(axes.flat)[len(selected):]:
                fig.delaxes(ax)
            save_figure(fig, out / "parity.png", dpi=300)
        finally:
            plt.close(fig)
    return statistics
