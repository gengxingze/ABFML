import sys
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    loss_file = sys.argv[1]

    data_pd = pd.read_csv(loss_file, sep=r'\s+', header=0)

    label_list = ["Loss", "E_tot", "Force", "Ei", "Virial"]
    for label in label_list:
        if "T_RMSE_" + label in data_pd.keys():
            plt.plot(data_pd["Epoch"], data_pd["T_RMSE_" + label], label='train')
            plt.plot(data_pd["Epoch"], data_pd["V_RMSE_" + label], label='valid')
            plt.xlabel('epoch')
            plt.ylabel(label)
            plt.grid(linestyle='--')
            plt.legend()
            plt.savefig(f'loss_{label}.png', dpi=128)
            plt.close()
