# -*- coding: utf-8 -*-

"""
Code from: https://github.com/gohsyi/cluster_optimization
"""

############################### Import libraries ###############################

from argparser import args
from env import Env
import os
import numpy as np
from tqdm import tqdm
from greedy import Greedy
from opt import Optimal
from learn_acer import Acer
from learn_ppo import learn_ppo
from plot import power_plot, latency_plot
if __name__ == '__main__':

    env = Env()
    models = [
        
        learn_ppo(),
        
        Acer(),

        Greedy(act_size=args.n_servers, n_servers=args.n_servers),
        
        Optimal(act_size=args.n_servers, n_servers=args.n_servers, tasks = env.tasks),
       
        
    ]

    
    for m in models:
        done = False
        obs = env.reset()
        for _ in tqdm(range(2000)):
            action = int(m.step(obs))
            _, _, done, info = env.step(action)

        latency, power = info
        np.savetxt(os.path.join('logs', f'{m.name}_Latency.txt'), latency)
        np.savetxt(os.path.join('logs', f'{m.name}_Power.txt'), power)
    
    power_plot()
    latency_plot()

