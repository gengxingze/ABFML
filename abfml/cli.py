"""ABFML command-line entry point."""

from __future__ import annotations

from abfml.data.reader import to_ase

import argparse
from collections.abc import Sequence

from abfml.logger import configure_logging
from abfml.workflows.train import run


def _plot_parser(parser):
    sub = parser.add_subparsers(dest="plot_kind", required=True)
    for name in ("labels", "predictions", "pairs", "neighbors", "descriptors"):
        command = sub.add_parser(name)
        if name == "pairs":
            command.add_argument("--output", help="CSV filename; default: LABEL.csv")
        else:
            command.add_argument("--output", nargs="?", const=".", default=None, metavar="DIR",
                                 help="Export numerical files; optional directory defaults to current directory")
        if name in {"predictions", "pairs"}:
            command.add_argument("input")
        if name == "predictions":
            command.add_argument("--labels", nargs="+")
        if name == "pairs":
            command.add_argument("--label", required=True)
        if name in ("labels", "neighbors"):
            command.add_argument("files", nargs="*" if name == "labels" else "+")
        if name in ("labels", "descriptors"):
            command.add_argument("--dataset", action="append", required=name == "descriptors",
                                 metavar="NAME=FILE", help="Repeat for groups/files")
        if name == "labels":
            command.add_argument("--bins", type=int, default=60)
        if name == "neighbors":
            command.add_argument("--cutoff", type=float, nargs="+", required=True)
            command.add_argument("--slots", type=int)
        if name == "descriptors":
            command.add_argument("--model", required=True)
            command.add_argument("--batch-size", type=int, default=8)
            command.add_argument("--level", choices=("atom", "structure"), default="atom",
                                 help="Analyze atomic environments or mean-pooled structure descriptors")
    return parser


def _run_plot(args):
    from abfml import tools
    from abfml.tools.common import print_table
    from pathlib import Path

    output = args.output or "."
    export = args.output is not None
    rows, columns = None, None
    groups = {}
    for item in getattr(args, "dataset", None) or []:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError("--dataset must be NAME=FILE")
        groups.setdefault(name, []).append(path)
    if args.plot_kind == "labels":
        if bool(args.files) == bool(groups):
            raise ValueError("Provide files or --dataset NAME=FILE, exclusively")
        rows = tools.plot_labels(groups or args.files, output=output, bins=args.bins, export=export)
        columns = {"Dataset": "dataset", "Label": "label", "Unit": "unit", "Count": "count",
                   "Min": "min", "Max": "max", "Mean": "mean", "Std": "std"}
    elif args.plot_kind == "predictions":
        rows = tools.plot_predictions(args.input, output=output, labels=args.labels)
        columns = {"Dataset": "dataset", "Label": "label", "Unit": "unit", "Count": "count", "RMSE": "rmse", "MAE": "mae"}
    elif args.plot_kind == "pairs":
        destination = args.output or f"{args.label}.csv"
        tools.export_pairs(args.input, label=args.label, output=destination)
        print(f"Paired values: {Path(destination).resolve()}")
    elif args.plot_kind == "neighbors":
        rows = tools.plot_neighbors(args.files, cutoffs=args.cutoff, output=output, slots=args.slots, export=export)
        columns = {"Cutoff (Å)": "cutoff", "Structures": "structures", "Centers": "count", "Min": "min",
                   "Max": "max", "Mean": "mean", "P95": "p95", "Overflow (%)": "overflow_percent"}
    elif args.plot_kind == "descriptors":
        rows = tools.plot_descriptors(args.model, groups, output=output, batch_size=args.batch_size,
                                      export=export, level=args.level)
        columns = {"Dataset": "dataset", "Level": "level", "Structures": "structures", "Centers": "centers",
                   "Points": "points", "Channels": "channels", "PC1 (%)": "pc1_percent", "PC2 (%)": "pc2_percent"}
    if rows:
        print_table(rows, columns)
        filename = {"labels": "labels.png", "neighbors": "neighbors.png", "descriptors": "descriptors.png", "predictions": "parity.png"}[args.plot_kind]
        print(f"Figure: {(Path(output)/filename).resolve()}")
        if export and args.plot_kind in {"labels", "neighbors", "descriptors"}:
            print(f"Numerical files: {Path(output).resolve()}")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser; epochs denotes additional epochs for this run."""
    parser = argparse.ArgumentParser(prog="abfml", description="ABFML machine learning force field toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _plot_parser(subparsers.add_parser("plot", help="Analyze labels, predictions, neighbors, and descriptors"))

    train = subparsers.add_parser("train", help="Train, resume, or fine-tune a model")
    train.add_argument("config", help="JSON configuration file")
    source = train.add_mutually_exclusive_group()
    source.add_argument("--resume", metavar="CHECKPOINT", help="Restore the complete training state")
    source.add_argument("--fine-tune", metavar="CHECKPOINT", help="Load model weights only")
    train.add_argument("--epochs", type=int, help="Override the number of additional epochs for this run")
    train.add_argument("--log-file", help="Also append logs to a UTF-8 file")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate independently and save per-structure predictions")
    evaluate.add_argument("model", help="Standard model file or checkpoint")
    evaluate.add_argument("files", nargs="+", help="Structure files supported by ASE")
    evaluate.add_argument("--output", default="evaluation", help="Evaluation output directory")
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--log-file", help="Also append logs to a UTF-8 file")
    evaluate.add_argument("--labels", nargs="+", help="Supervised labels required in every sample")

    check = subparsers.add_parser("check", help="Check physical consistency; generate a test structure when omitted")
    check.add_argument("model")
    check.add_argument("structure", nargs="?", help="Optional structure file; otherwise generate a structure from model species")
    check.add_argument("--index", type=int, default=0)
    check.add_argument("--device", default="cpu")
    check.add_argument("--log-file", help="Also append logs to a UTF-8 file")
    check.add_argument("--step", type=float, default=1.e-5)
    check.add_argument("--atol", type=float, default=1.e-5)
    check.add_argument("--rtol", type=float, default=1.e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse and execute command-line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(args, "log_file", None))
    if args.command == "plot":
        try:
            _run_plot(args)
        except ValueError as error:
            parser.error(str(error))
    elif args.command == "train":
        run(
            args.config,
            resume=args.resume,
            fine_tune=args.fine_tune,
            epochs=args.epochs,
        )
    elif args.command == "evaluate":
        from abfml.workflows.evaluate import evaluate
        evaluate(args.model, args.files, output_dir=args.output,
                 batch_size=args.batch_size, device=args.device, labels=args.labels)
    elif args.command == "check":
        from abfml.workflows.check import check_model
        from abfml.data import ASEReader
        atoms = None
        if args.structure is not None:
            atoms = to_ase(next(iter(ASEReader(args.structure, index=args.index))))
        report = check_model(args.model, atoms, device=args.device, index=args.index,
                             step=args.step, atol=args.atol, rtol=args.rtol)
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
