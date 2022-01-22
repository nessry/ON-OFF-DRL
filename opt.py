# -*- coding: utf-8 -*-
############################### Import libraries ###############################

"""
Define the optimal solution for the MIP model
"""

from argparser import args
import gurobipy as grb
from time import sleep

class Optimal(object):
    def __init__(self, act_size, n_servers, tasks):
        self.name = 'optimal'       # set model name
        self.n_servers = n_servers
        self.act_size = act_size
        self.tasks = tasks
        self.P_0 = args.P_0
        self.P_100 = args.P_100

    def step(self, obs):
        opt_model = grb.Model(name="MIP Model")     # define MIP model using gurobipy package
        self.latency = [t.start_time - t.arrive_time for t in self.tasks]
        for i in range(1, len(self.latency)):
            self.latency[i] = self.latency[i] + self.latency[i - 1]
        # Define the binary decision variable x for users association
        x_vars  = {(i):opt_model.addVar(vtype=grb.GRB.BINARY,
                        name="x_" + str(i)) 
                        for i in range(self.n_servers)}
        # add model constraints
        constraints = {i : 
                       opt_model.addConstr(
                           lhs=grb.quicksum(x_vars[i] for i in range(self.n_servers)),
                           sense=grb.GRB.LESS_EQUAL,
                           rhs=1, 
                           name="constraint_{0}".format(i))for i in range(self.n_servers)}
        # set model objective function
        objective = grb.quicksum(x_vars[i] * (self.P_0 + (self.P_100 - self.P_0) * (2 * obs[i << 1] - obs[i << 1]**(1.4))) 
                         for i in range(self.n_servers)) + grb.quicksum(x_vars[i] *self.latency[i] for i in range(self.n_servers))
        # for minimization
        opt_model.ModelSense = grb.GRB.MINIMIZE
        opt_model.setObjective(objective)
        opt_model.optimize()    # optimize model and get solutions
        for v in opt_model.getVars():
            if v.x > 1e-6:
                print(v.varName[2])
                sleep(0.05)
                result = v.varName[2]
        return result