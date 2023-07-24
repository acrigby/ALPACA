# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 09:03:56 2023

@author: localuser
"""

import random
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch import nn
from collections import deque,namedtuple
import glob
import io
import base64
import os

from dymola.dymola_interface import DymolaInterface
from Agent_PyTorch import *

model = "NHES.Systems.Examples.TES_Use_Case.HTGR_Case_01_IndependentBOP_Uprated200MW_Wpumps"
#initial = "C:/Users/aidan/OneDrive - UW-Madison/PhD/Dymola Output/Initial40K.txt"

# Instantiate the Dymola interface and start Dymola
dymola = DymolaInterface()
print(dymola.DymolaVersion())

dymola.openModel("C:/Users/localuser/HYBRID/Models/NHES/package.mo")
dymola.openModel("C:/Users/localuser/HYBRID/TRANSFORM-Library/TRANSFORM-Library/TRANSFORM/package.mo")
dymola.AddModelicaPath("C:/Users/localuser/Documents/Dymola")


dymola = DymolaInterface()
print(dymola.DymolaVersion())
dymola.openModel(model)

env = gym.make('AcrobotDymola', render_mode = "human")
device = torch.device("cpu")

learningtimes = []
learningtimesav = []
iterations = []
window = 5
i = 0

class ReplayMemory(object):

    def __init__(self, capacity):
        # Define a queue with maxlen "capacity"
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        # Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self)) # Get all the samples if the requested batch_size is higher than the number of sample currently in the memory
        # Randomly select "batch_size" samples and return the selection
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory) # Return the number of samples currently stored in the memory
    
class DQN(nn.Module):

    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()

        self.linear = nn.Sequential(
                  nn.Linear(state_space_dim,64),
                  nn.ReLU(),
                  nn.Linear(64,64*2),
                  nn.ReLU(),
                  nn.Linear(64*2,action_space_dim)
                )

    def forward(self, x):
        x = x.to(device)
        return self.linear(x)

### Define exploration profile
initial_value = 5
num_iterations = 800
exp_decay = np.exp(-np.log(initial_value) / num_iterations * 6) # We compute the exponential decay in such a way the shape of the exploration profile does not depend on the number of iterations
exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)

### PARAMETERS
gamma = 0.99   # gamma parameter for the long term reward
replay_memory_capacity = 10000   # Replay memory capacity
#lr = 1e-2   # Optimizer learning rate
#lr = 1e-4
lr = 1e-3
target_net_update_steps = 10   # Number of episodes to wait before updating the target network
batch_size = 256   # Number of samples to take from the replay memory for each update
bad_state_penalty = 0   # Penalty to the reward when we are in a bad state (in this case when the pole falls down) 
min_samples_for_training = 1000   # Minimum samples in the replay memory to enable the training

# Get the shapes of the state space (observation_space) and action space (action_space)
state_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.n
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"STATE SPACE SIZE: {state_space_dim}")
print(f"ACTION SPACE SIZE: {action_space_dim}")

### Initialize the replay memory
replay_mem = ReplayMemory(replay_memory_capacity)    

### Initialize the policy network
policy_net = DQN(state_space_dim, action_space_dim).to(device)

### Initialize the target network with the same weights of the policy network
target_net = DQN(state_space_dim, action_space_dim).to(device)
target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

### Initialize the optimizer
optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr) # The optimizer will update ONLY the parameters of the policy network

### Initialize the loss function (Huber loss)
loss_fn = nn.SmoothL1Loss()

# Initialize the Gym environment
env = gym.make('AcrobotDymola', render_mode = "human")
#env = TimeLimit(env, max_episode_steps=5000)
observation, info = env.reset()

plotting_rewards=[]

for episode_num, tau in enumerate(tqdm(exploration_profile)):

    # Reset the environment and get the initial state
    observation, info = env.reset()
    # Reset the score. The final score will be the total amount of steps before the pole falls
    score = 0
    terminated = False
    truncated = False

    # Go on until the pole falls off
    while not (terminated or truncated):

      # Choose the action following the policy
      action, q_values = choose_action_softmax(policy_net, observation, temperature=tau)
      
      # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
      next_observation, reward, terminated, truncated, info = env.step(action)

      # Update the final score (+1 for each step)
      score += reward

      # Apply penalty for bad state
      if terminated or truncated: # if the pole has fallen down 
          reward += bad_state_penalty
          next_observation = None
      
      # Update the replay memory
      replay_mem.push(observation, action, next_observation, reward)

      # Update the network
      if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
          update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size, device)

      observation = next_observation

    # Update the target network every target_net_update_steps episodes
    if episode_num % target_net_update_steps == 0:
        print('Updating target network...')
        target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network
    
    plotting_rewards.append(score)
    # Print the final score
    print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau}") # Print the final score
    
    # i += 1
    # plt.plot(env.results["Time"],env.results["revolute1.phi"])
    # plt.plot(env.results["Time"],env.results["revolute2.phi"])
    # plt.show()
    # print(env.t)
    # learningtimes.append(env.t)
    # iterations.append(i)
    # if i < window:
    #     learningtimesav.insert(0,np.nan)
    # else:
    #     learningtimesav.append(np.mean(learningtimes[i-window:i]))

    # plt.plot(iterations,learningtimes, 'k.-', label='Original data')
    # plt.plot(iterations,learningtimesav, 'r.-', label='Running average')
    # plt.yscale('log')
    # plt.grid(linestyle=':')
    # plt.legend()
    # plt.show()

env.close()

plt.plot(learningtimes, 'k.-', label='Original data')
plt.plot(learningtimesav, 'r.-', label='Running average')
plt.show()