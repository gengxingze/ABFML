import argparse
from typing import List, Optional, Union
from abfml.entrypoints.train import train_mlff
from abfml.entrypoints.valid import valid_mlff
from abfml.entrypoints.check import check_mlff

def main_command(args: Optional[Union[List[str], argparse.Namespace]] = None):
    dict_args = vars(args)
    if args.command == "train":
        train_mlff(**dict_args)
    elif args.command == "valid":
        valid_mlff(**dict_args)
    elif args.command == "check":
        check_mlff(**dict_args)
    else:
        raise Exception(f"undefined command")
