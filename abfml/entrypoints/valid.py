import json
import torch
import numpy as np
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
    unit = {'energy': 'eV', 'force':'eV/$\AA$', 'virial': 'eV'}
    # Print the Quality Of Data header
    logger.info(f"|{'Quality Of Data'.center(94)}|")
    logger.info(f"| {'Key':<16} {'DFT Mean':>17} {'DFT Sigma':>17} {'Predict Mean':>17} {'Predict Sigma':>17}     |")
    logger.info(f"+----------------------------------------------------------------------------------------------+")

    # Store evaluation metrics
    indicators = []

    for predict_key in predict_data.keys():
        if len(predict_data[predict_key]) != 0:
            # Concatenate data
            dft = np.concatenate([arr[predict_key].detach().numpy().flatten() for arr in valid_data_load])
            predict = np.concatenate([arr.detach().numpy().flatten() for arr in predict_data[predict_key]])
            difference = dft - predict

            # Statistical data
            dft_mean, dft_sigma = np.mean(dft), np.std(dft)
            predict_mean, predict_sigma = np.mean(predict), np.std(predict)

            # Calculate evaluation metrics
            mse = np.mean(difference ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(difference))
            r2 = 1 - (np.sum(difference ** 2) / np.sum((dft - dft_mean) ** 2))

            # Output Quality Of Data
            logger.info(
                f"| {predict_key:<16} {dft_mean:>17.4e} {dft_sigma:>17.4f} {predict_mean:>17.4e} {predict_sigma:>17.4f}     |")

            # Save metrics data for later use
            indicators.append((predict_key, rmse, mse, mae, r2))

            # Plot scatter plot (DFT vs Predict)
            if plot:
                from scipy.stats import gaussian_kde
                import matplotlib.pyplot as plt
                plt.rcParams.update({'font.size': 18, 'font.family': 'serif', 'font.serif': ['Times New Roman']})
                plt.rcParams['mathtext.default'] = 'regular'
                plt.figure(figsize=(10, 8))  # Increase image size
                plt.scatter(dft, predict, s=20, label='data')  # Enhance scatter plot effect

                # Diagonal line (y = x, ideal fit line)
                data_min = min(dft.min(), predict.min())
                data_max = max(dft.max(), predict.max())
                plt.plot([data_min, data_max], [data_min, data_max], color='black', linestyle='--', linewidth=1.5,
                         label='$y=x$')

                # Annotate RMSE and R² values
                plt.text(0.05, 0.95, f"RMSE: {rmse:4.4f}{unit[predict_key]}\nR²: {r2:4.4f}", transform=plt.gca().transAxes,
                         fontsize=18, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

                # Set title, axis labels, and grid
                plt.title(f'{predict_key}: dft vs predict', fontsize=22)
                plt.xlabel(f'{predict_key}$_{{dft}}$', fontsize=20)
                plt.ylabel(f'{predict_key}$_{{predict}}$', fontsize=20)
                plt.xlim(data_min, data_max)
                plt.ylim(data_min, data_max)
                plt.tick_params(axis='both', labelsize=18)
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.legend()

                # Save scatter plot
                plt.savefig(f'{predict_key}_scatter.png', dpi=300, bbox_inches='tight')
                plt.close()

                # Plot error distribution (Difference Histogram + KDE)
                plt.figure(figsize=(10, 8))

                # Histogram part
                counts, bins, _ = plt.hist(difference, bins=40, density=True, alpha=0.7, color='#1f77b4',
                                           edgecolor='black')

                # KDE (Kernel Density Estimation) smoothed density curve
                kde = gaussian_kde(difference)
                kde_x = np.linspace(bins.min(), bins.max(), 300)
                plt.plot(kde_x, kde(kde_x), color='red', linewidth=2, label='Density')

                # Annotate RMSE and R² values
                plt.text(0.65, 0.95, f"RMSE: {rmse:4.4f}{unit[predict_key]}\nR²: {r2:4.4f}", transform=plt.gca().transAxes,
                         fontsize=18, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

                # Set title, axis labels, and grid
                plt.title(f'{predict_key} error distribution', fontsize=22)
                plt.xlabel('error (dft - predict)', fontsize=20)
                plt.ylabel('density', fontsize=20)
                plt.tick_params(axis='both', labelsize=18)
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.legend()

                # Save error distribution plot
                plt.savefig(f'ErrorDistributionOf_{predict_key}.png', dpi=300, bbox_inches='tight')
                plt.close()

                # Save data (DFT and Predict)
                np.savetxt(f'{predict_key}.csv', np.column_stack((dft, predict)), header='DFT,Predict', delimiter=',')

    logger.info(f"+----------------------------------------------------------------------------------------------+")

    # Print the Evaluation Indicators header
    logger.info(f"|{'Evaluation Indicators'.center(94)}|")
    logger.info(f"| {'Key':<16} {'RMSE':>17} {'MSE':>17} {'MAE':>17} {'R²':>17}     |")
    logger.info(f"+----------------------------------------------------------------------------------------------+")

    # Output metrics data
    for predict_key, rmse, mse, mae, r2 in indicators:
        logger.info(f"| {predict_key:<16} {rmse:>17.4e} {mse:>17.4e} {mae:>17.4e} {r2:>17.4f}     |")

    logger.info(f"+----------------------------------------------------------------------------------------------+")
