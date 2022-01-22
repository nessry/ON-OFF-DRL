# -*- coding: utf-8 -*-
############################### Import libraries ###############################

import os
import pandas as pd
import matplotlib.pyplot as plt

    


fig_num = 0     # change this to prevent overwriting figures in same env_name folder

# smooth out rewards to get a smooth (window_len ++) or a less smooth (window_len --) plot lines, you can change the window size to change the curve format
window_len_var = 5
min_window_len_var = 1
linewidth_var = 2.5
alpha_var = 1

def ppo_plot():
    log_dir = "PPO_files"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + 'resource_allocation' + '/' + 'stability' + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    colors = ['red', 'orange', 'black']
    print("============================================================================================")

    # make directory for saving figures
    figures_dir = "plots"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # make environment directory for saving figures
    figures_dir = figures_dir + '/' + 'resource_allocation' + '/' + 'stability' + '/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)


    fig_save_path = figures_dir + '/PPO_' + 'resource_allocation' + '_fig_' + str(fig_num) + '.pdf'


    # get number of log files in directory
    log_dir = "PPO_files" + '/' + 'resource_allocation' + '/' + 'stability' + '/'

    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)

    all_runs = []

    for run_num in range(num_runs):

        log_f_name = log_dir + '/PPO_' + 'resource_allocation' + "_log_" + str(run_num) + ".csv"
        print("loading data from : " + log_f_name)
        data = pd.read_csv(log_f_name)
        data = pd.DataFrame(data)
    
        all_runs.append(data)
        print("--------------------------------------------------------------------------------------------")


    ax = plt.gca()

    for i, run in enumerate(all_runs):
        # smooth out rewards to get a smooth or a less smooth (var) plot lines
        run['reward_var_' + str(i)] = run['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()
        # plot the lines
        run.plot(kind='line', x='timestep' , y='reward_var_' + str(i),ax=ax,color=colors[i % len(colors)],  linewidth=linewidth_var, alpha=alpha_var, marker ='^',markevery=10)
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.legend(['PPO_32','PPO_64','PPO_256'], fontsize=11, loc='lower right')
    fig = plt.gcf()
    fig.tight_layout()
    plt.savefig(fig_save_path, dpi=600, facecolor='w', edgecolor='b',
    orientation='landscape', papertype=None, format=None,
    transparent=True, bbox_inches=None, pad_inches=0.1,
    metadata=None)
    print("figure saved at : ", fig_save_path)
    plt.show()
    print("============================================================================================")

def acer_plot():
    log_dir = "ACER_files"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + 'resource_allocation' + '/' + 'stability' + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    colors = ['green', 'purple', 'blue']
    print("============================================================================================")

    # make directory for saving figures
    figures_dir = "plots"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # make environment directory for saving figures
    figures_dir = figures_dir + '/' + 'resource_allocation' + '/' + 'stability' + '/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)


    fig_save_path = figures_dir + '/ACER_' + 'resource_allocation' + '_fig_' + str(fig_num) + '.pdf'


    # get number of log files in directory
    log_dir = "ACER_files" + '/' + 'resource_allocation' + '/' + 'stability' + '/'

    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)

    all_runs = []

    for run_num in range(num_runs):

        log_f_name = log_dir + '/ACER_' + 'resource_allocation' + "_log_" + str(run_num) + ".csv"
        print("loading data from : " + log_f_name)
        data = pd.read_csv(log_f_name)
        data = pd.DataFrame(data)
    
        all_runs.append(data)
        print("--------------------------------------------------------------------------------------------")

    ax = plt.gca()

    for i, run in enumerate(all_runs):
        # smooth out rewards to get a smooth or a less smooth (var) plot lines
        run['reward_var_' + str(i)] = run['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()
        # plot the lines
        run.plot(kind='line', x='timestep' , y='reward_var_' + str(i),ax=ax,color=colors[i % len(colors)],  linewidth=linewidth_var, alpha=alpha_var,marker ='o',markevery=10)
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.legend(['ACER_32','ACER_64','ACER_256'], fontsize=11, loc='lower right')
    fig = plt.gcf()
    fig.tight_layout()
    plt.savefig(fig_save_path, dpi=600, facecolor='w', edgecolor='b',
    orientation='landscape', papertype=None, format=None,
    transparent=True, bbox_inches=None, pad_inches=0.1,
    metadata=None)
    print("figure saved at : ", fig_save_path)
    plt.show()
    print("============================================================================================")
     
if __name__ == '__main__':
    ppo_plot()
    acer_plot()
