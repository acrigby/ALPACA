# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:07:13 2023

@author: localuser
"""

import random
import torch
import numpy as np
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch import nn
from collections import deque,namedtuple
from dymola.dymola_interface import DymolaInterface
import glob
import io
import base64
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
def choose_action_softmax(net, state, temperature):
    
    if temperature < 0:
        raise Exception('The temperature value must be greater than or equal to 0 ')
        
    # If the temperature is 0, just select the best action using the eps-greedy policy with epsilon = 0
    if temperature == 0:
        return choose_action_epsilon_greedy(net, state, 0)
    
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32)
        net_out = net(state)

    # Apply softmax with temp
    temperature = max(temperature, 1e-8) # set a minimum to the temperature for numerical stability
    softmax_out = nn.functional.softmax(net_out/temperature, dim=0).cpu().numpy()
                
    # Sample the action using softmax output as mass pdf
    all_possible_actions = np.arange(0, softmax_out.shape[-1])
    
    print(softmax_out)
    
    print(all_possible_actions)
    
    # this samples a random element from "all_possible_actions" with the probability distribution p (softmax_out in this case)
    action = np.random.choice(all_possible_actions,p=softmax_out)
    
    print(action)
    return action, net_out.cpu().numpy()

policy_net = DQN(4, 3).to(device) 

action, q_values = choose_action_softmax(policy_net, [10000,1,1,1], temperature=5)