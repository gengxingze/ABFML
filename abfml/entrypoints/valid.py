import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader, Subset
from typing import Optional
from abfml.logger.loggers import Logger, log_data_info, log_logo
from abfml.data.read_data import ReadData
from abfml.train.trainer import valid_loop


def valid_mlff(
        *,
        model: str,
        numb_test: int,
        plot: bool,
        shuffle: bool,
        datafile: Optional[list[str]],
        **kwargs):
    logger = Logger("valid.log").logger
    log_logo(logger=logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_threads = torch.get_num_threads()
    num_worker = int(num_threads / 4)

    if ".ckpt" in model:
        logger.info(f"| Because file {model} has the suffix ckpt,")
        logger.info(f"| we will attempt to read the model from a checkpoint-type file.")
        model_class = torch.load(model)["model"]
    else:
        logger.info(f"| We will attempt to read the model from a jit_script-type file {model}.")
        model_class = torch.jit.load(model)
    model_class.to(device=device)

    if 'valid.input' in datafile[0] and len(datafile) == 1:
        with open('valid.input', 'r') as file:
            input_json = json.load(file)
            filename_list = input_json['valid_file']
    else:
        filename_list = datafile

    logger.info("+-------------------------------------- valid data file ---------------------------------------+")
    valid_dataclass = ReadData(filename=filename_list,
                               cutoff=model_class.cutoff,
                               neighbor=model_class.neighbor,
                               type_map=model_class.type_map,
                               file_format=None)
    log_data_info(logger, valid_dataclass)
    valid_data = ConcatDataset(valid_dataclass.get_mlffdata())
    total_indices = np.arange(len(valid_data))
    if shuffle:
        logger.info(f"| You will randomly select  {numb_test:>4d} image")
        np.random.shuffle(total_indices)
    else:
        logger.info(f"| You will use the first {numb_test:>4d} image")
    logger.info(f"+----------------------------------------------------------------------------------------------+")
    valid_indices = total_indices[:numb_test]
    subset = Subset(valid_data, valid_indices)
    valid_data_load = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    _, predict_data = valid_loop(data_load=valid_data_load,
                                 model=model_class,
                                 logger_name="valid.log",
                                 print_freq=1,
                                 save_predict=True)
    logger.info(f"+----------------------------------------------------------------------------------------------+")
    summaries_valid_dft = "|-- DFT: sigma(StandardDeviation)"
    summaries_valid_predict = "|-- Predict: sigma(StandardDeviation)"
    summaries_valid_difference = "|-- Difference: sigma(StandardDeviation)"
    for predict_key in predict_data.keys():
        if len(predict_data[predict_key]) != 0:
            dft = np.concatenate([arr[predict_key].detach().numpy().flatten() for arr in valid_data_load])
            predict = np.concatenate([arr.detach().numpy().flatten() for arr in predict_data[predict_key]])
            difference = dft - predict

            summaries_valid_dft += (f"\n        |      {predict_key:<15s}: "
                                    f"sigma = {np.std(dft):>2.4f} , mean = {np.mean(dft):>.4e} ")
            summaries_valid_predict += (f"\n        |      {predict_key:<15s}: "
                                        f"sigma = {np.std(predict):>2.4f} , mean = {np.mean(predict):>.4e} ")
            summaries_valid_difference += (f"\n        |      {predict_key:<15s}: "
                                           f"sigma = {np.std(difference):>2.4f} , mean = {np.mean(difference):>.4e} , "
                                           f"RMSE = {np.sqrt(np.mean(difference ** 2)):>.4e}")
            if plot:
                plt.scatter(dft, predict)
                data_min = min(dft.min(), predict.min())
                data_max = max(dft.max(), predict.max())
                plt.plot([data_min, data_max], [data_min, data_max], color='black', linestyle='--')
                plt.xlabel(predict_key + '$_DFT$')
                plt.ylabel(predict_key + '$_predict$')
                plt.xlim(data_min, data_max)
                plt.ylim(data_min, data_max)
                plt.savefig(predict_key + '.png', dpi=128)
                plt.close()

                counts, bins, _ = plt.hist(difference, bins=20, density=False, alpha=0.7)
                plt.clf()
                frequencies = counts / np.sum(counts)
                plt.bar(bins[:-1], frequencies * 100.0, width=np.diff(bins), alpha=0.7)
                plt.xlabel('Error')
                plt.ylabel('Density(%)')
                plt.title(f'Distribution of {predict_key} error')
                plt.savefig(f'ErrorDistributionOf{predict_key}.png', dpi=128)
                plt.close()

                np.savetxt(predict_key + '.csv', np.column_stack((dft, predict)), header='dft, predict')

    logger.info(summaries_valid_dft)
    logger.info(summaries_valid_predict)
    logger.info(summaries_valid_difference)
    logger.info(f"+----------------------------------------------------------------------------------------------+")
