import argparse
import textwrap
from typing import List, Optional


class RawTextArgumentDefaultsHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """This formatter is used to print multile-line help message with default value."""


def main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ABFML: A package for rapid building, fitting, and application of machine learning force fields",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="ABFML-kit subcommands", dest="command")

    # **************************************   train parser start **************************************
    parser_train = subparsers.add_parser(
        name="train",
        parents=[],
        help="train a model",
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """\
        examples:
            abfml train input.json
            abfml train input.json --restart model.ckpt
            abfml train input.json --init-model model.ckpt
        """
        ),
    )
    parser_train.add_argument(
        "INPUT", help="the input parameter file in json format."
    )
    parser_train_subgroup = parser_train.add_mutually_exclusive_group()
    parser_train_subgroup.add_argument(
        "-i",
        "--init_model",
        type=str,
        default=None,
        help="Initialize the model by the provided checkpoint.",
    )
    parser_train_subgroup.add_argument(
        "-r",
        "--restart",
        type=str,
        default=None,
        help="Initialize the training from the frozen model.",
    )
    # **************************************   train parser end   **************************************

    # **************************************   valid parser start **************************************
    parser_valid = subparsers.add_parser(
        name="valid",
        parents=[],
        help="valid the model",
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """\
        examples:
            abfml valid -m model.pt -f /path/to/system -n 30
        """
        ),
    )
    parser_valid.add_argument(
        "-m",
        "--model",
        default="model.pt",
        type=str,
        help="Valid model file to import",
    )
    parser_valid.add_argument(
        "-s",
        "--shuffle",
        default=False,
        action="store_true",
        help="Shuffle data and randomised",
    )
    parser_valid.add_argument(
        "-f",
        "--datafile",
        default=None,
        nargs="+",
        type=str,
        help="The path to file of test list.",
    )
    parser_valid.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="information",
    )
    parser_valid.add_argument(
        "-n", "--numb_test",
        default=100,
        type=int,
        help="The number of data for test"
    )

    # **************************************   valid parser end   **************************************

    # **************************************   check parser start **************************************
    parser_check = subparsers.add_parser(
        "check",
        parents=[],
        help="check the model",
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent(
            """\
        examples:
            abfml check -m model.pt 
            abfml check -i input.json -d float32
        """
        ),
    )

    parser_check_subgroup = parser_check.add_mutually_exclusive_group()
    parser_check_subgroup.add_argument(
        "-i",
        "--input",
        default=None,
        type=str,
        help="the input parameter file in json format.",
    )
    parser_check_subgroup.add_argument(
        "-m",
        "--model_name",
        default=None,
        type=str,
        help="check model file to import",
    )
    parser_check.add_argument(
        "-d",
        "--dtype",
        default="float32",
        type=str,
        help="Accuracy used when checking the model",
    )
    # ************************************** check parser end   **************************************

    return parser


def main_parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments and convert argument strings to objects.

    Parameters
    ----------
    args : List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv

    Returns
    -------
    argparse.Namespace
        the populated namespace
    """
    parser = main_parser()
    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()
    return parsed_args
