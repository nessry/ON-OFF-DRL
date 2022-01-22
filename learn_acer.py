# -*- coding: utf-8 -*-
############################### Import libraries ###############################

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from env import Env
from argparser import args
from collections import deque


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class EpisodicReplayMemory(object):
    def __init__(self, capacity, max_episode_length):
        self.num_episodes = capacity // max_episode_length
        self.buffer = deque(maxlen=self.num_episodes)
        self.buffer.append([])
        self.position = 0
        
    def push(self, state, action, reward, policy, mask, done):
        self.buffer[self.position].append((state, action, reward, policy, mask))
        if done:
            self.buffer.append([])
            self.position = min(self.position + 1, self.num_episodes - 1)
            
    def sample(self, batch_size, max_len=None):
        min_len = 0
        while min_len == 0:
            rand_episodes = random.sample(self.buffer, batch_size)
            min_len = min(len(episode) for episode in rand_episodes)
            
        if max_len:
            max_len = min(max_len, min_len)
        else:
            max_len = min_len
            
        episodes = []
        for episode in rand_episodes:
            if len(episode) > max_len:
                rand_idx = random.randint(0, len(episode) - max_len)
            else:
                rand_idx = 0

            episodes.append(episode[rand_idx:rand_idx+max_len])
            
        return list(map(list, zip(*episodes)))
    
    def __len__(self):
        return len(self.buffer)
    
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions)
        )
        
        
    def forward(self, x):
        policy  = self.actor(x).clamp(max=1-1e-20)
        q_value = self.critic(x)
        value   = (policy * q_value).sum(-1, keepdim=True)
        return policy, q_value, value
    
def compute_acer_loss(policies, q_values, values, actions, rewards, retrace, masks, behavior_policies, gamma=0.99, truncation_clip=10, entropy_weight=0.0001):
    loss = 0
    
    for step in reversed(range(len(rewards))):
        importance_weight = policies[step].detach() / behavior_policies[step].detach()

        retrace = rewards[step] + gamma * retrace * masks[step]
        advantage = retrace - values[step]

        log_policy_action = policies[step].gather(1, actions[step]).log()
        truncated_importance_weight = importance_weight.gather(1, actions[step]).clamp(max=truncation_clip)
        actor_loss = -(truncated_importance_weight * log_policy_action * advantage.detach()).mean(0)

        correction_weight = (1 - truncation_clip / importance_weight).clamp(min=0)
        actor_loss -= (correction_weight * policies[step].log() * (q_values[step] - values[step]).detach()).sum(1).mean(0)
        
        entropy = entropy_weight * -(policies[step].log() * policies[step]).sum(1).mean(0)

        q_value = q_values[step].gather(1, actions[step])
        critic_loss = ((retrace - q_value) ** 2 / 2).mean(0)

        truncated_rho = importance_weight.gather(1, actions[step]).clamp(max=1)
        retrace = truncated_rho * (retrace - q_value.detach()) + values[step].detach()
        
        loss += actor_loss + critic_loss - entropy
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
def load(model, checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    
def off_policy_update(batch_size, replay_ratio=4):
    if batch_size > len(replay_buffer) + 1:
        return
    
    for _ in range(np.random.poisson(replay_ratio)):
        trajs = replay_buffer.sample(batch_size)
        state, action, reward, old_policy, mask = map(torch.stack, zip(*(map(torch.cat, zip(*traj)) for traj in trajs)))

        q_values = []
        values   = []
        policies = []

        for step in range(state.size(0)):
            policy, q_value, value = model(state[step])
            q_values.append(q_value)
            policies.append(policy)
            values.append(value)

        _, _, retrace = model(state[-1])
        retrace = retrace.detach()
        compute_acer_loss(policies, q_values, values, action, reward, retrace, mask, old_policy)
        
torch.manual_seed(0)
# preTrained weights directory
random_seed = 0             #### set this to load a particular checkpoint trained on random seed
run_num_pretrained = 0      #### set this to load a particular checkpoint num
directory = "ACER_preTrained" + '/' + 'resource_allocation' + '/' 
checkpoint_path = directory + "ACER256_{}_{}_{}.pth".format('resource_allocation', random_seed, run_num_pretrained)
print("loading network from : " + checkpoint_path)
model = ActorCritic(args.n_servers * args.n_resources + args.n_resources + 1, args.n_servers).to(device)
#model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
optimizer = optim.Adam(model.parameters())
capacity = 10000
max_episode_length = 200
replay_buffer = EpisodicReplayMemory(capacity, max_episode_length)
env=Env()
max_ep_len = 200
state = env.reset()

class Acer(object):
    
    def __init__(self):
        self.name = 'ACER'
        
    def step(self, obs):
        state = obs
        done = False
        total_reward = 0
        for t in range(1, max_ep_len+1):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy, _, _ = model(state)
            action = policy.multinomial(1)
            next_state, reward, done, _ = env.step(action.item())
            state = next_state
            total_reward += reward
            if done:
                break
    
        return action

