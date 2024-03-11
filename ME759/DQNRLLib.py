# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 08:09:44 2024

@author: aidan
"""

from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.tune.logger import pretty_print

from dymola.dymola_interface import DymolaInterface
import matplotlib.pyplot as plt

# Note: `gymnasium` (not `gym`) will be **the** API supported by RLlib from Ray 2.3 on.
import gymnasium as gym

import os

import torch
import numpy as np
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch import nn
from dymola.dymola_interface import DymolaInterface
from csv import writer
import os
from SolverClasses import *
import logging

print(os.path.abspath(os.curdir))
os.chdir("..")
print(os.path.abspath(os.curdir))
ALPACApath = os.path.abspath(os.curdir)

f = open("./Utilities/ALPACAASCII.txt", 'r')
print(''.join([line for line in f]))

#register new enviroments
exec(open(str(ALPACApath)+"/gymnasium/gymnasium/envs/classic_control/__init__.py").read())

env_name = "Reactor-v4"
env = gym.make(env_name)
dymola2 = DymolaInterface()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

#define orginial results profile for plotting
Orig_results = {}
Orig_variables = ["Time","sensor_pT.T"]    
for key in Orig_variables:
    Orig_results[key] = []
Orig_trajsize = dymola2.readTrajectorySize(str(ALPACApath)+"/RunData/Original_Temp_Profile.mat")
Orig_signals = dymola2.readTrajectory(str(ALPACApath)+"/RunData/Original_Temp_Profile.mat", Orig_variables, Orig_trajsize)

for i in range(0,len(Orig_variables),1):
    Orig_results[Orig_variables[i]].extend(Orig_signals[i])
    
for i in range(0,len(Orig_results["Time"]),1):
    Orig_results["Time"][i] = Orig_results["Time"][i] - 9900


algo = (
    DQNConfig().training(lr=0.005)
    .rollouts(num_rollout_workers=4)
    .resources(num_gpus=0)
    .environment(env= "Reactor-v4")
    .build()
)

for i in range(30):
    result = algo.train()
    print(pretty_print(result))
    observation, info = env.reset()
    
    # Reset the score. The final score will be the total amount of steps before the pole falls
    score = 0
    t = 0
    temps = []
    mdots = []
    powers = []
    TCVdps = []
    feeds = []
    t1 = []
    terminated = False
    truncated = False
    
    temps.append(observation[0]+673.15)
    feeds.append(observation[3])
    t1.append(t*5)

    # Go on until the pole falls off
    while not (terminated or truncated):
      t+=1

      action = algo.compute_single_action(observation)
      next_observation, reward, terminated, truncated, info = env.step(action)
      score += reward
      
      temps.append(next_observation[0]+673.15)
      feeds.append(next_observation[3])
      t1.append(t*5)
         
      #print(reward)

      # Update the final score (+1 for each step)
      #score += reward

      # Apply penalty for bad state
      if terminated or truncated: # if the pole has fallen down 
          next_observation = None
      
      observation = next_observation
    
    fig, axs = plt.subplots(2,1)
    axs[0].plot(Orig_results["Time"],Orig_results["sensor_pT.T"], label = 'Original')
    axs[0].plot(t1, temps)
    axs[0].axhspan(671.15, 675.15, color='green', alpha=0.5, label = "Temperature Band")
    axs[0].set_xlim(0,500)
    axs[0].legend()
    axs[0].set_title('Temperature')
    # axs[0, 1].plot(t1, mdots, 'tab:orange')
    # axs[0, 1].set_title('Pump Mdot')
    axs[1].plot(t1, feeds, 'tab:green')
    axs[1].set_title('FF Signal')
    # axs[1, 1].plot(t1, TCVdps, 'tab:red')
    # axs[1, 1].set_title('TCV pressure drop')
    fig.tight_layout()
    axs[1].set(xlabel='Time / s')
        
    display.display(plt.gcf())
    plt.close()
    # Print the final score
    print(f"FINAL SCORE: {score} - Temperature: {0}") # Print the final score
        


    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")
