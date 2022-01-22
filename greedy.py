# -*- coding: utf-8 -*-
"""
Code from: https://github.com/gohsyi/cluster_optimization
"""

"""
Define the greedy solution that associates the task to the server with the lowest CPU usage.
"""

############################### Import libraries ###############################
import numpy as np


class Greedy(object):
    def __init__(self, act_size, n_servers):
        self.name = 'greedy'        # set model name
        self.n_servers = n_servers
        self.act_size = act_size

    def step(self, obs):
        m_cpu = (100, [])
        # find the server with the lowest CPU usage
        for i in range(self.n_servers):
            cpu = obs[i << 1]
            if cpu < m_cpu[0]:
                m_cpu = (cpu, [i])
            elif cpu == m_cpu[0]:
                m_cpu[1].append(i)
        return np.random.choice(m_cpu[1])
