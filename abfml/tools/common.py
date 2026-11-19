"""Shared internal plotting, output-directory, and statistical-summary helpers."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCIENCE_STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "axes.linewidth": .8, "legend.fontsize": 9,
}


def style_axes(ax):
    """Scientific plotting style with four spines, inward ticks, and no grid."""
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=4, width=.8)
    ax.tick_params(which="minor", length=2, width=.6)

def write_table(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def save_figure(fig, path, *, dpi=180):
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=.1)
    plt.close(fig)

def summarize(values):
    values = np.asarray(values).ravel()
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite values")
    return dict(count=values.size, mean=values.mean(), std=values.std(),
                min=values.min(), p05=np.quantile(values, .05),
                median=np.median(values), p95=np.quantile(values, .95), max=values.max())


def output_dir(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def print_table(rows, columns):
    """Print statistics; columns maps display headings to result fields."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}"
        # Windows GBK cannot encode angstrom symbols; use A in terminals and the symbol in figures.
        return str(value).replace("Å", "A")
    values = [[cell(name) for name in columns]]
    values.extend([cell(row.get(key, "N/A")) for key in columns.values()] for row in rows)
    widths = [max(len(row[index]) for row in values) for index in range(len(columns))]
    lines = [" | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in values]
    lines.insert(1, "-+-".join("-"*width for width in widths))
    print("\n".join(lines))
