"""
Code from: https://github.com/gohsyi/cluster_optimization
"""

############################### Import libraries ###############################
import argparse

parser = argparse.ArgumentParser()

# environment setting
parser.add_argument('-n_servers', type=int, default=10)
parser.add_argument('-n_resources', type=int, default=2)
parser.add_argument('-n_tasks', type=int, default=2000,
                    help='Use all tasks by default.')
parser.add_argument('-w1', type=float, default=1e-4)
parser.add_argument('-w2', type=float, default=1e-4)
parser.add_argument('-w3', type=float, default=1e-4)
parser.add_argument('-P_0', type=int, default=87)
parser.add_argument('-P_100', type=int, default=145)
parser.add_argument('-T_on', type=int, default=30)
parser.add_argument('-T_off', type=int, default=30)
args = parser.parse_args()





