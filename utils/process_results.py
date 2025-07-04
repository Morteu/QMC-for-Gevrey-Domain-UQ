import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from domain_generation.reference_domain import Ellipse
from domain_generation.perturbation import *

def import_and_append(folder_path):
    folder_path = os.path.expanduser(folder_path)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    all_results = pd.concat(dataframes, ignore_index=True)
    return all_results

def calculate_rmse(df):
    def get_rmse(col):
        mean = col.mean()
        q = len(col)
        mse = ((col - mean)**2).sum() / (q * (q - 1))
        rmse = np.sqrt(mse)
        return rmse
    
    s = df.shape[1]
    rmse_df = pd.DataFrame(columns=['method', 'n', 'rmse'])
    for method in df.Method.unique():
        for n in df.n.unique():
            rmse = df[(df.Method == method) & (df.n == n)].iloc[:, 3:s].apply(get_rmse).mean()
            rmse_df.loc[len(rmse_df)] = [method, n, rmse]

    rmse_df = rmse_df.sort_values(['n', 'method'])
    return rmse_df

def plot_results(df, threshold=0):
    """
    Plot the results and perform polyfit only on n values above the given threshold.
    
    Args:
    df (pd.DataFrame): The dataframe containing the results (method, n, rmse).
    threshold (int): The minimum value of n to consider for polyfit. Default is 0 (i.e., use all values).
    """

    # Define style
    # plt.style.use('ggplot')
    # plt.style.use('bmh')

    plt.figure(figsize=(8,8))

    colors = {'Generic QMC':['green', 'darkgreen'], 'Weight-dependent QMC':['red', 'brown'], 'MC':['blue', 'darkblue']}

    methods = ['MC', 'Generic QMC', 'Weight-dependent QMC']
    for method in methods:
        if method == "Generic QMC":
            method_name = "QMC, off-the-shelf $z$"
        else:
            method_name = method

        ns = df[df.method == method].n
        rmses = df[df.method == method].rmse

        # Filter n values above the threshold
        ns_filtered = ns[ns > threshold]
        rmses_filtered = rmses[ns > threshold]

        # Perform polyfit only on the filtered data
        log_ns = np.log10(ns_filtered)
        log_rmses = np.log10(rmses_filtered)
        slope, intercept = np.polyfit(log_ns, log_rmses, 1)
        fit_line = 10**(intercept + slope * log_ns)

        # Plot all points
        plt.loglog(ns, rmses, 'o', color=colors[method][0], label=f'{method_name} (slope={slope:.2f})')

        # Plot the regression line
        plt.loglog(ns_filtered, fit_line, '--', color=colors[method][1])   

    plt.xlabel('$n$', fontsize=20, labelpad=10)
    plt.ylabel('rms error estimate', fontsize=20, labelpad=10)
    plt.ylim(0,0.15)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(fontsize=20, loc='upper center', facecolor=(1, 1, 1, 0), frameon=False)
    save_path = os.path.expanduser("~/Desktop/Convergence.eps")
    plt.savefig(save_path, format='eps', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def plot_reconstruction(folder, n, Dref):
    file = folder + f'/results_par_n{n}.csv'

    df = pd.read_csv(file)

    s = df.shape[1]
    domain = Ellipse(num_points=128, a=1, b=1)
    x_boundary, y_boundary = domain.get_boundary()

    plt.figure(figsize=(8, 8))

    colors = {"Weight-dependent QMC": "red", "MC": "blue"}

    for method in ["Weight-dependent QMC", "MC"]:
        df_new = df[df.Method == method]        
        new_ys = df_new.iloc[:, 3:s].mean()
        Dw_boundary = reconstruct_V_exp(x_boundary, y_boundary, new_ys, s-3)
        if method == "MC":
            plt.plot(Dw_boundary[0], Dw_boundary[1], label=method, color=colors[method], linewidth=5, linestyle='dotted')
        else:
            plt.plot(Dw_boundary[0], Dw_boundary[1], label=method, color=colors[method], linewidth=2)

    if Dref == 'square':
        side = 1.8
        half_side = side/2
        x_square = np.array([-half_side, half_side, half_side, -half_side, -half_side])
        y_square = np.array([-half_side, -half_side, half_side, half_side, -half_side])
        plt.plot(x_square, y_square, color='blue')
        disk = Rectangle((-half_side, -half_side), side, side, color='green')
        plt.gca().add_patch(disk)
    
    elif Dref == 'sampled':
        og_y = np.load('inputs/y_sample.npy')
        original_boundary_x, original_boundary_y = get_V_exp(x_boundary, y_boundary, og_y, len(og_y))
        # plt.fill(original_boundary_x, original_boundary_y, color='lightsteelblue')
        plt.plot(original_boundary_x, original_boundary_y, color='gray', label='Original domain', linewidth=2)

    # plt.title(f'Reconstruction for n = {n}', fontsize=16)
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,2])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20, facecolor=(1, 1, 1, 0), frameon=False)
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18)

    file_path = os.path.expanduser(f'~/Desktop/Reconstruction_QMC_n{n}.eps')
    plt.savefig(file_path, format='eps', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def double_reconstruction(folder, n):
    file = folder + f'/results_par_n{n}.csv'
    df = pd.read_csv(file)

    s = df.shape[1]
    domain = Ellipse(num_points=128, a=1, b=1)
    x_boundary, y_boundary = domain.get_boundary()

    colors = {"Weight-dependent QMC": "red", "MC": "green"}

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for the first method: Weight-dependent QMC
    method = "Weight-dependent QMC"
    df_new = df[df.Method == method]
    new_ys = df_new.iloc[:, 3:s].mean()
    Dw_boundary = reconstruct_V_exp(x_boundary, y_boundary, new_ys, s - 3)
    axes[0].plot(Dw_boundary[0], Dw_boundary[1], label=method, color=colors[method])

    og_y = np.load('inputs/y_sample.npy')
    original_boundary_x, original_boundary_y = get_V_exp(x_boundary, y_boundary, og_y, len(og_y))
    axes[0].fill(original_boundary_x, original_boundary_y, color='lightsteelblue')
    axes[0].plot(original_boundary_x, original_boundary_y, color='blue', label='Original domain')

    axes[0].set_xlim([-1.5, 1.5])
    axes[0].set_ylim([-1.5, 1.75])
    axes[0].tick_params(axis='x', labelsize=14)  
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].legend(fontsize=16)
    axes[0].set_xlabel("$x_1$", fontsize=16)
    axes[0].set_ylabel("$x_2$", fontsize=16)

    # Plot for the second method: MC
    method = "MC"
    df_new = df[df.Method == method]
    new_ys = df_new.iloc[:, 3:s].mean()
    Dw_boundary = reconstruct_V_exp(x_boundary, y_boundary, new_ys, s - 3)
    axes[1].plot(Dw_boundary[0], Dw_boundary[1], label=method, color=colors[method])

    axes[1].fill(original_boundary_x, original_boundary_y, color='lightsteelblue')
    axes[1].plot(original_boundary_x, original_boundary_y, color='blue', label='Original domain')

    axes[1].set_xlim([-1.5, 1.5])
    axes[1].set_ylim([-1.5, 1.75])
    axes[1].tick_params(axis='x', labelsize=14)  
    axes[1].tick_params(axis='y', labelsize=14)
    axes[1].legend(fontsize=16)
    axes[1].set_xlabel("$x_1$", fontsize=16)
    axes[1].set_ylabel("$x_2$", fontsize=16)

    # Save and display the figure
    file_path = os.path.expanduser(f'~/Desktop/Reconstruction_n{n}.eps')
    plt.savefig(file_path, format='eps', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def calculate_delta_differences():
    # TODO: calculate the difference in the delta values in the reconstruction, to see if QMC is better than MC when visually they look similar
    pass
