import argparse
from typing import List, Optional, Union
from abfml.entrypoints.train import train_mlff
from abfml.entrypoints.valid import valid_mlff


def main_command(args: Optional[Union[List[str], argparse.Namespace]] = None):
    dict_args = vars(args)
    if args.command == "train":
        print("train")
        train_mlff(**dict_args)
    elif args.command == "valid":
        print("valid")
        valid_mlff(**dict_args)
    elif args.command == "predict":
        print("predict")
    else:
        raise Exception(f"undefined command")
