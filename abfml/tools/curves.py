"""Training loss and metric curves."""
from pathlib import Path
from .common import plt, np, output_dir, SCIENCE_STYLE, style_axes

def plot_loss(path, *, output=".", log=False):
    """Plot metrics in one figure, sharing axes for training and validation of each label."""
    out = output_dir(output)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    columns = next((line.split(":", 1)[1].split() for line in lines if line.startswith("# columns:")), [])
    units = next((line.split(":", 1)[1].split() for line in lines if line.startswith("# units:")), [])
    if "Epoch" not in columns or len(units) != len(columns):
        raise ValueError("loss.out requires matching columns/units headers and an Epoch column")
    data = np.array([[float(x) if x != "NA" else np.nan for x in line.split()]
                     for line in lines if line.strip() and not line.startswith("#")])
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError("Empty or malformed loss.out")
    panels = {}
    titles = {"loss": "Loss", "energy": "Energy RMSE", "force": "Force RMSE",
              "virial": "Virial RMSE", "lr": "Learning rate", "time": "Epoch time"}
    for i, name in enumerate(columns):
        if name in {"Epoch", "time"} or not np.isfinite(data[:, i]).any():
            continue
        if name in {"loss", "V-loss"}:
            key, label = "loss", "Train" if name == "loss" else "Valid"
        elif name.startswith(("T-", "V-")):
            key, label = name[2:], "Train" if name.startswith("T-") else "Valid"
        else:
            key, label = name, None
        panels.setdefault(key, []).append((i, label))
    if not panels:
        raise ValueError("loss.out contains no finite metrics")
    # Show training quality before learning rate; inspect loss.out for raw values.
    ordered = [key for key in panels if key not in {"lr", "time"}]
    ordered += [key for key in ("lr", "time") if key in panels]
    panels = {key: panels[key] for key in ordered}

    ncols = min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    with plt.rc_context(SCIENCE_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.9 * nrows),
                                 squeeze=False, layout="constrained")
        try:
            epochs = data[:, columns.index("Epoch")]
            for panel, (key, series) in enumerate(panels.items()):
                ax = axes.flat[panel]
                panel_units = {units[i] for i, _ in series}
                if len(panel_units) != 1:
                    raise ValueError(f"Inconsistent train/valid units for {key}")
                unit = next(iter(panel_units))
                for i, label in series:
                    ax.plot(epochs, data[:, i], label=label,
                            color="#D55E00" if label == "Valid" else "#0072B2",
                            linestyle="--" if label == "Valid" else "-", linewidth=1.5)
                values = data[:, [i for i, _ in series]]
                finite = values[np.isfinite(values)]
                # Use log axes only for strictly positive metrics; never hide zero or negative values.
                if log and key != "time" and (finite > 0).all():
                    ax.set_yscale("log")
                title = titles.get(key, f"{key.replace('_', ' ').capitalize()} RMSE"
                                   if any(label for _, label in series) else key)
                ax.set_title(f"({chr(97 + panel)})  {title}", loc="left", pad=10)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(title if unit == "-" else f"{title} ({unit})")
                style_axes(ax)
                ax.margins(x=.02)
                if any(label for _, label in series):
                    ax.legend(frameon=False)
            for ax in list(axes.flat)[len(panels):]:
                fig.delaxes(ax)
            fig.savefig(out / "loss.png", dpi=300)
        finally:
            plt.close(fig)
