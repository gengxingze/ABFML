import logging
from abfml.data.read_data import ReadData


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name: str, fmt: str = ".4e", summary_type: str = ""):
        self.name = name
        self.fmt = fmt
        summary_type_list = ["NONE", "AVERAGE", "SUM", "COUNT"]
        if summary_type in summary_type_list:
            self.summary_type = summary_type
        else:
            raise ValueError(f"Invalid summary type : {summary_type}, supported only {summary_type_list}")
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = f"{self.name}: {self.val:{self.fmt}}"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type == "NONE":
            fmtstr = f"{self.name}: {self.val:{self.fmt}}"
        elif self.summary_type == "AVERAGE":
            fmtstr = f"{self.name}: {self.avg:{self.fmt}}"
        elif self.summary_type == "SUM":
            fmtstr = f"{self.name}: {self.sum:{self.fmt}}"
        elif self.summary_type == "COUNT":
            fmtstr = f"{self.name}: {self.count:{self.fmt}}"
        else:
            raise ValueError("Invalid summary type %r" % self.summary_type)
        return fmtstr


# Set to singleton mode, only one logger is allowed to exist for the same inputs
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    def __init__(self, log_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(filename=log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter(fmt='[%(levelname)s]- %(message)s', datefmt='%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)


def log_logo(logger:logging.Logger):
    logger.info(r"################################################################################################")
    logger.info(r"#                         _       ______   ________  ____    ____  _____                       #")
    logger.info(r"#                        / \     |_   _ \ |_   __  ||_   \  /   _||_   _|                      #")
    logger.info(r"#                       / _ \      | |_) |  | |_ \_|  |   \/   |    | |                        #")
    logger.info(r"#                      / ___ \     |  __'.  |  _|     | |\  /| |    | |   _                    #")
    logger.info(r"#                    _/ /   \ \_  _| |__) |_| |_     _| |_\/_| |_  _| |__/ |                   #")
    logger.info(r"#                   |____| |____||_______/|_____|   |_____||_____||________|                   #")
    logger.info(r"#                                                                                              #")
    logger.info(r"################################################################################################")


def log_data_info(logger: logging.Logger, dataclass: ReadData):
    logger.info("| ################ file name ############### Atom number ## nframes ######### Element ######## |")
    total_frames = 0

    for data_info in dataclass.data_information:
        file_name = data_info.get("file_name", "")[-45:]
        n_frames = data_info.get("n_frames", 0)
        elements = data_info.get("include_element", [])
        atom_counts = data_info.get("include_atoms_number", set())

        symbols = ','.join(elements) if len(elements) <= 5 else "complex"
        atom_number = str(next(iter(atom_counts))) if len(atom_counts) == 1 else "variation"

        logger.info(f"|{file_name:>49s}{atom_number:>6s}{n_frames:>8d}{symbols:>18s}")
        total_frames += n_frames

    logger.info(f"| Number of files found: {len(dataclass.data_information):>4d}, number of images: {total_frames:>5d}")
    logger.info(f"+----------------------------------------------------------------------------------------------+")

