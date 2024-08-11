import os
import sys
from abfml.entrypoints.main_parser import main_parse_args
from abfml.entrypoints.main_command import main_command
from abfml.entrypoints.train import train_mlff
from abfml.entrypoints.valid import valid_mlff


def main():
    print("__main__")
    args = main_parse_args()
    main_command(args=args)


if __name__ == '__main__':
    print("__main__")
    train_mlff(INPUT=r"D:\Work\PyCharm\ABFML\example\new-MLFF\usedefined_input.json", init_model=None, restart=None)
    # valid_mlff(model=r"D:\work\Pycharm\mlff\test\checkpoint_final.ckpt", numb_test=100,plot=False,
    #            shuffle=False,
    #            datafile=r"D:\work\Pycharm\mlff\test\data\Cu_128_1200_vasprun.xml")