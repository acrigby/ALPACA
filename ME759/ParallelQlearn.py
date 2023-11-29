# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:29:29 2023

@author: aidan
"""

import random
import torch
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from torch import nn
from collections import deque,namedtuple
import glob
import io
import base64
import os
import multiprocessing
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from SolverClasses import *
import time
start_time = time.time()
     
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


### Create environment
env = gym.make('Acrobot') # Initialize the Gym environment

# Get the shapes of the state space (observation_space) and action space (action_space)
state_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

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
env = gym.make('Acrobot') 
observation, info = env.reset()

plotting_rewards=[]

for episode_num, tau in enumerate(exploration_profile):

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
      if score % 5 == 1:
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

env.close()

plt.plot(plotting_rewards)
plt.savefig('learn.png')

# Initialize the Gym environment
env = gym.make('Acrobot') 
observation, info = env.reset()
plotting_rewards_final = []

for episode_num in range(10):

    # Reset the environment and get the initial state
    observation, info = env.reset()
    # Reset the score. The final score will be the total amount of steps before the pole falls
    score = 0
    terminated = False
    truncated = False

    # Go on until the pole falls off
    while not (terminated or truncated):

      # Choose the action following the policy
      action, q_values = choose_action_softmax(policy_net, observation, temperature=0)
      
      # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
      next_observation, reward, terminated, truncated, info = env.step(action)

      # Update the final score (+1 for each step)
      score += reward

      # Apply penalty for bad state
      if terminated or truncated: # if the pole has fallen down 
          reward += bad_state_penalty
          next_observation = None
      

      observation = next_observation

    plotting_rewards_final.append(score)
    # Print the final score
    print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau}") # Print the final score

env.close()

plt.plot(plotting_rewards_final)
plt.savefig('final.png')

print("--- %s seconds ---" % (time.time() - start_time))