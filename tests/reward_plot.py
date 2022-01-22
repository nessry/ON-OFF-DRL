# -*- coding: utf-8 -*-
############################### Import libraries ###############################

import os
import pandas as pd
import matplotlib.pyplot as plt


# smooth out rewards to get a smooth (window_len ++) or a less smooth (window_len --) plot lines, you can change the window size to change the curve format
window_len_var = 5
min_window_len_var = 1
linewidth_var = 2.5
alpha_var = 1

def step_plot():
    fig_num = 0     # change this to prevent overwriting figures in same env_name folder
    colors = ['green', 'red']
     # make directory for saving figures
    figures_dir = "plots"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # make environment directory for saving figures
    figures_dir = figures_dir + '/' + 'resource_allocation' + '/' + 'reward' + '/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    fig_save_path = figures_dir + '/step_reward'  + '_fig_' + str(fig_num) + '.pdf'
    
    log_dir = "ACER_files"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + 'resource_allocation' + '/' + 'reward' + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    # get number of log files in directory
    log_dir = "ACER_files" + '/' + 'resource_allocation' + '/' + 'reward' + '/'

    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)

    all_runs = []

    for run_num in range(num_runs):

        log_f_name = log_dir + '/ACER_' + 'resource_allocation' + "_log_" + str(run_num) + ".csv"
        print("loading data from : " + log_f_name)
        data = pd.read_csv(log_f_name)
        data = pd.DataFrame(data)
    
        all_runs.append(data)
    ax = plt.gca()

    for i, run in enumerate(all_runs):
        # smooth out rewards to get a smooth or a less smooth (var) plot lines
        run['reward_var_' + str(i)] = run['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()
        # plot the lines
        run.plot(kind='line', x='timestep' , y='reward_var_' + str(i),ax=ax,color='green',  linewidth=linewidth_var, alpha=alpha_var,marker ='o',markevery=10)
    print("============================================================================================")
    
    log_dir = "PPO_files"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + 'resource_allocation' + '/' + 'reward' + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    # get number of log files in directory
    log_dir = "PPO_files" + '/' + 'resource_allocation' + '/' + 'reward' + '/'

    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)

    all_runs = []

    for run_num in range(num_runs):

        log_f_name = log_dir + '/PPO_' + 'resource_allocation' + "_log_" + str(run_num) + ".csv"
        print("loading data from : " + log_f_name)
        data = pd.read_csv(log_f_name)
        data = pd.DataFrame(data)
    
        all_runs.append(data)
    ax = plt.gca()

    for i, run in enumerate(all_runs):
        # smooth out rewards to get a smooth or a less smooth (var) plot lines
        run['reward_var_' + str(i)] = run['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()
        # plot the lines
        run.plot(kind='line', x='timestep' , y='reward_var_' + str(i),ax=ax,color='red',  linewidth=linewidth_var, alpha=alpha_var, marker ='^',markevery=10)
    print("============================================================================================")
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.legend(['ACER','PPO'], fontsize=11, loc='lower right')
    plt.annotate('ACER convergence point', xy =(18500, -2.21),
             xytext =(20395, -2.8),bbox = dict(boxstyle ="round", fc ="0.8"),
                arrowprops = dict(
                    arrowstyle = "->",
                    connectionstyle = "angle, angleA = 0, angleB = 90,\
                        rad = 100"),)
    plt.annotate('PPO convergence point', xy =(22147, -1.891),
             xytext =(3000, -1.75),bbox = dict(boxstyle ="round", fc ="0.8"),
                arrowprops = dict(
                    arrowstyle = "->",
                    connectionstyle = "angle, angleA = 0, angleB = 120,\
                        rad = 10"),)
    fig = plt.gcf()
    fig.tight_layout()
    plt.savefig(fig_save_path, dpi=600, facecolor='w', edgecolor='b',
    orientation='landscape', papertype=None, format=None,
    transparent=True, bbox_inches=None, pad_inches=0.1,
    metadata=None)
    print("figure saved at : ", fig_save_path)
    plt.show()
    print("============================================================================================")

def episode_plot():
    fig_num = 0     # change this to prevent overwriting figures in same env_name folder
    colors = ['green', 'red']
     # make directory for saving figures
    figures_dir = "plots"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # make environment directory for saving figures
    figures_dir = figures_dir + '/' + 'resource_allocation' + '/' + 'reward' + '/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    fig_save_path = figures_dir + '/episode_reward'  + '_fig_' + str(fig_num) + '.pdf'
    
    log_dir = "ACER_files"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + 'resource_allocation' + '/' + 'reward' + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    # get number of log files in directory
    log_dir = "ACER_files" + '/' + 'resource_allocation' + '/' + 'reward' + '/'

    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)

    all_runs = []

    for run_num in range(num_runs):

        log_f_name = log_dir + '/ACER_' + 'resource_allocation' + "_log_" + str(run_num) + ".csv"
        print("loading data from : " + log_f_name)
        data = pd.read_csv(log_f_name)
        data = pd.DataFrame(data)
    
        all_runs.append(data)
    ax = plt.gca()

    for i, run in enumerate(all_runs):
        # smooth out rewards to get a smooth or a less smooth (var) plot lines
        run['reward_var_' + str(i)] = run['reward'].rolling(window=1, win_type='triang', min_periods=min_window_len_var).mean()
        # plot the lines
        run.plot(kind='line', x='episode' , y='reward_var_' + str(i),ax=ax,color='green',  linewidth=linewidth_var, alpha=alpha_var)
    print("============================================================================================")
    
    log_dir = "PPO_files"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + 'resource_allocation' + '/' + 'reward' + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    # get number of log files in directory
    log_dir = "PPO_files" + '/' + 'resource_allocation' + '/' + 'reward' + '/'

    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)

    all_runs = []

    for run_num in range(num_runs):

        log_f_name = log_dir + '/PPO_' + 'resource_allocation' + "_log_" + str(run_num) + ".csv"
        print("loading data from : " + log_f_name)
        data = pd.read_csv(log_f_name)
        data = pd.DataFrame(data)
    
        all_runs.append(data)
    ax = plt.gca()

    for i, run in enumerate(all_runs):
        # smooth out rewards to get a smooth or a less smooth (var) plot lines
        run['reward_var_' + str(i)] = run['reward'].rolling(window=1, win_type='triang', min_periods=min_window_len_var).mean()
        # plot the lines
        run.plot(kind='line', x='episode' , y='reward_var_' + str(i),ax=ax,color='red',  linewidth=linewidth_var, alpha=alpha_var)
    print("============================================================================================")
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_xlabel("Episodes", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.legend(['ACER','PPO'], fontsize=11, loc='lower right')
    plt.xlim(0,350)
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
    step_plot()
    episode_plot()


