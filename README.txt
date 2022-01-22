- This is the code for the paper: On-Policy vs. Off-Policy Deep Reinforcement Learning for Resource Allocation in Open Radio Access Network

- It has been tested on Windows 10 and Python 3.8.3

- To run the code, mainly you need:
 * pip install torch
 * pip install gurobipy

- To avoid the long training time, you could go directly to the tests folder and run:
 * reward_plot.py to get the step and episode reward figures (first two figures)
 * stability_plot.py to get the ACER and PPO reward figures for different NN architectures
 * plot.py to get the energy and energy per latency figures (last two figures)

- To start training from scratch, you need to generate the reward files and the trained models weights by running:
 * acer_32.py, then
 * acer_64.py, then
 * acer_256.py, then
 * ppo_32.py, then
 * ppo_64.py, then
 * ppo_256.py, then
 * reward_plot.py and stability_plot.py

As a result you will create folders (PPO_files, PPO_pretrained, ACER_files, ACER_pretrained) 
that contains reward files and trained models respectively.

- learn_acer.py and learn_ppo.py load the trained models to test them in energy and latency performance
 * run main.py to to do these tests and plot the energy and energy per latency figures.
- opt.py and greedy.py implement the optimal MIP solution and the greedy solution respectively.

----- REFERENCES -----
 * https://github.com/higgsfield/RL-Adventure-2
 * https://github.com/nikhilbarhate99/PPO-PyTorch
 * https://github.com/gohsyi/cluster_optimization