# -*- coding: utf-8 -*-
############################### Import libraries ###############################

import os
import numpy as np
import matplotlib.pyplot as plt

linewidth_smooth = 2.5
alpha_smooth = 1

fig_save_path1 = os.path.join('plots','power.pdf')
fig_save_path2 = os.path.join('plots','latency.pdf')


def power_plot():
    ax = plt.gca()

    list1 = np.loadtxt(os.path.join('logs', 'optimal_Power.txt'))
    list2 = np.loadtxt(os.path.join('logs', 'greedy_Power.txt'))
    list3 = np.loadtxt(os.path.join('logs', 'ACER_Power.txt'))
    list4 = np.loadtxt(os.path.join('logs', 'PPO_Power.txt'))
            
    ax.plot(list1,color = 'blue', linewidth=linewidth_smooth, alpha=alpha_smooth, marker ='+',markevery=100)
    ax.plot(list2,color = 'orange', linewidth=linewidth_smooth, alpha=alpha_smooth, marker ='s',markevery=100)
    ax.plot(list3,color ='green', linewidth=linewidth_smooth, alpha=alpha_smooth, marker ='o',markevery=100)
    ax.plot(list4,color = 'red',linewidth=linewidth_smooth, alpha=alpha_smooth, marker ='^',markevery=100)
            
    ax.set_xlabel('Number of network users', fontsize=11)
    ax.set_ylabel('Energy (kWh)', fontsize=11)
    ax.legend(['Optimal','Greedy','ACER','PPO'], fontsize=11, loc='lower right')
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.1)
    plt.xlim(1000, 2000)
    
    fig = plt.gcf()
    fig.tight_layout()
    plt.savefig(fig_save_path1, dpi=600, facecolor='w', edgecolor='b',
            orientation='landscape', papertype=None, format=None,
            transparent=True, bbox_inches=None, pad_inches=0.1,
                        metadata=None)
            
    plt.show()

def latency_plot():
    ax = plt.gca()

    list1 = np.loadtxt(os.path.join('logs', 'optimal_Power.txt'))
    list2 = np.loadtxt(os.path.join('logs', 'greedy_Power.txt'))
    list3 = np.loadtxt(os.path.join('logs', 'ACER_Power.txt'))
    list4 = np.loadtxt(os.path.join('logs', 'PPO_Power.txt'))
            
    list5 = np.loadtxt(os.path.join('logs', 'optimal_Latency.txt'))
    list6 = np.loadtxt(os.path.join('logs', 'greedy_Latency.txt'))
    list7 = np.loadtxt(os.path.join('logs', 'ACER_Latency.txt'))
    list8 = np.loadtxt(os.path.join('logs', 'PPO_Latency.txt'))
            
    ax.plot(list5,list1,color = 'blue', linewidth=linewidth_smooth, alpha=alpha_smooth, marker ='+',markevery=100)
    ax.plot(list6,list2,color = 'orange', linewidth=linewidth_smooth, alpha=alpha_smooth, marker ='s',markevery=100)
    ax.plot(list7,list3,color ='green', linewidth=linewidth_smooth, alpha=alpha_smooth, marker ='o',markevery=100)
    ax.plot(list8,list4,color = '#E50000',linewidth=linewidth_smooth, alpha=alpha_smooth, marker ='^',markevery=100)
            
    ax.set_xlabel('Latency (sec)', fontsize=11)
    ax.set_ylabel('Energy (kWh)', fontsize=11)
    ax.legend(['Optimal','Greedy','ACER','PPO'], fontsize=11, loc='upper left')
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.1)
    plt.xlim(300000, 1000000)
    
    fig = plt.gcf()
    fig.tight_layout()
    plt.savefig(fig_save_path2, dpi=600, facecolor='w', edgecolor='b',
            orientation='landscape', papertype=None, format=None,
            transparent=True, bbox_inches=None, pad_inches=0.1,
                        metadata=None)
            
    plt.show()

if __name__ == '__main__':
    power_plot()
    latency_plot()

