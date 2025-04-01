import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# Set font style to Times New Roman
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman']})
plt.rcParams['mathtext.default'] = 'regular'

def plot_loss_curve(file_name):
    """
    Reads a loss file and plots the training and validation RMSE curves.

    Parameters:
        file_name (str): Path to the input CSV file.
    """
    # Read the data file
    data_pd = pd.read_csv(file_name, sep=r'\s+', header=0)

    # List of RMSE metrics to be plotted
    label_list = ["Loss", "E_tot", "Force", "Ei", "Virial"]

    for label in label_list:
        train_key = f"T_RMSE_{label}"
        valid_key = f"V_RMSE_{label}"

        # Check if both training and validation keys exist in the dataset
        if train_key in data_pd and valid_key in data_pd:
            # Create a figure and axis with a larger size for better readability
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot training and validation RMSE curves
            ax.plot(data_pd["Epoch"], data_pd[train_key], label="Train", linestyle="-", marker="o",
                    markersize=4)
            ax.plot(data_pd["Epoch"], data_pd[valid_key], label="Valid", linestyle="--", marker="s",
                    markersize=4)

            # Set title and axis labels
            ax.set_title(f"{label} RMSE vs Epoch", fontsize=18)
            ax.set_xlabel("Epoch", fontsize=18)
            ax.set_ylabel(f"{label} RMSE", fontsize=18)
            ax.tick_params(axis='both', labelsize=16)

            # Add grid lines with transparency
            ax.grid(linestyle="--", alpha=0.6)

            # Improve legend visibility
            ax.legend(fontsize=18, loc="best", frameon=True, edgecolor="gray")

            # Save the figure as a high-resolution PNG file
            plt.savefig(f'loss_{label}.png', dpi=300, bbox_inches="tight")
            plt.close()


def plot_scatter(dft, predict, predict_key, rmse, r2, unit):
    """
    Plot scatter diagram comparing DFT and predicted values.
    """
    plt.figure(figsize=(10, 8))  # Increase image size
    plt.scatter(dft, predict, s=20, label='Data', alpha=0.6)  # Enhance scatter plot effect

    # Diagonal line (y = x, ideal fit line)
    data_min = min(dft.min(), predict.min())
    data_max = max(dft.max(), predict.max())
    plt.plot([data_min, data_max], [data_min, data_max], color='black', linestyle='--', linewidth=1.5, label='$y=x$')

    # Annotate RMSE and R² values
    plt.text(0.05, 0.95, f"RMSE: {rmse:.4f}{unit}\nR²: {r2:.4f}", transform=plt.gca().transAxes,
             fontsize=18, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    # Set title, axis labels, and grid
    plt.title(f'{predict_key}: DFT vs Predict', fontsize=22)
    plt.xlabel(f'{predict_key}$_{{DFT}}$', fontsize=20)
    plt.ylabel(f'{predict_key}$_{{Predict}}$', fontsize=20)
    plt.xlim(data_min, data_max)
    plt.ylim(data_min, data_max)
    plt.tick_params(axis='both', labelsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Save scatter plot
    plt.savefig(f'{predict_key}_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_distribution(difference, predict_key, rmse, r2, unit):
    """
    Plot error distribution using histogram and KDE.
    """
    plt.figure(figsize=(10, 8))

    # Histogram part
    counts, bins, _ = plt.hist(difference, bins=40, density=True, alpha=0.7, color='#1f77b4', edgecolor='black')

    # KDE (Kernel Density Estimation) smoothed density curve
    kde = gaussian_kde(difference)
    kde_x = np.linspace(bins.min(), bins.max(), 300)
    plt.plot(kde_x, kde(kde_x), color='red', linewidth=2, label='Density')

    # Annotate RMSE and R² values
    plt.text(0.65, 0.95, f"RMSE: {rmse:.4f}{unit}\nR²: {r2:.4f}", transform=plt.gca().transAxes,
             fontsize=18, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    # Set title, axis labels, and grid
    plt.title(f'{predict_key} Error Distribution', fontsize=22)
    plt.xlabel('Error (DFT - Predict)', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.tick_params(axis='both', labelsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Save error distribution plot
    plt.savefig(f'ErrorDistributionOf_{predict_key}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Ensure the script is run with a valid input file
    if len(sys.argv) < 2:
        print("Usage: python script.py <loss_file>")
        sys.exit(1)

    # Get the loss file from command-line arguments
    loss_file = sys.argv[1]
    plot_loss_curve(loss_file)
