# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 08:09:44 2024

@author: aidan
"""

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm
# Note: `gymnasium` (not `gym`) will be **the** API supported by RLlib from Ray 2.3 on.
import gymnasium as gym

import os
import sys
import torch
import numpy as np
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch import nn
from csv import writer
import os
import logging

sys.path.append("/home/rigbac/Projects/ALPACA/HPC_runs/PythonScripts")
from matreader import matreader
from PriceReader import importdict
print(os.path.abspath(os.curdir))
#os.chdir("..")
print(os.path.abspath(os.curdir))
ALPACApath = os.path.abspath(os.curdir)

f = open("./Utilities/ALPACAASCII.txt", 'r')
print(''.join([line for line in f]))

#register new enviroments
#exec(open(str(ALPACApath)+"/gymnasium/gymnasium/envs/classic_control/__init__.py").read())

env_name = "ReactorEcom-v11"
env = gym.make(env_name)

os.chdir('/home/rigbac/Projects/ALPACA')

"""
algo = (
    SACConfig()
    .env_runners(num_env_runners=20, num_envs_per_env_runner=2)
    .training(train_batch_size=(512),lr=0.001)
    .resources(num_cpus_per_worker=1,num_gpus=0)
    .environment(env= "ReactorEcom-v2")
    .build()
)
"""
algo = (
    PPOConfig()
    .env_runners(num_env_runners=80,rollout_fragment_length=1)
    .training(train_batch_size=(320), lr_schedule = [[0,1e-4],[200000,1e-8]])
    .resources(num_cpus_per_worker=0.5)
    .environment(env= "ReactorEcom-v11")
    .build()
)



#algo = Algorithm.from_checkpoint("/home/rigbac/ray_results/PPO_2024-05-22_16-22-51/PPO_Reactor-v8_d6998_00000_0_lr=0.0010_2024-05-22_16-22-56/checkpoint_000072")

rewards = []

for i in range(3000):
    
    
    result = algo.train()
    print(pretty_print(result))
    
    if i % 50 == 0:
        
        observation, info = env.reset()
    
        # Reset the score. The final score will be the total amount of steps before the pole falls
        score = 0
        t = 1003600
        powers = []
        price_list = []
        demands = []
        times = []
        scorelist = [0]
        tscore = [t]

        terminated = False
        truncated = False
        
    
        # Go on until the pole falls off
        while not (terminated or truncated):
          print(observation)
          #print(observation[-3])
          action, state, extra = algo.compute_single_action(observation, full_fetch = True, explore=False)
          print(f'Action = {action} : Prob = {extra[SampleBatch.ACTION_PROB]}')
          #print(action)
          next_observation, reward, terminated, truncated, results = env.step(action)
          times.append(t/(3600*24))
          times.append((t+3599)/(3600*24))
          price_list.append(next_observation[-1])
          price_list.append(next_observation[-1])
          score += (reward)*1000
          t+=3600

          tscore.append(t/(3600*24))
          scorelist.append(score)
             
          #print(reward)
    
          # Update the final score (+1 for each step)
          #score += reward
    
          # Apply penalty for bad state
          if terminated or truncated: # if the pole has fallen down 
              next_observation = None
          
          observation = next_observation
        
        rewards.append(score)

        #price_schedule,prices = importdict('/home/rigbac/Projects/ALPACA/HPC_runs/Input/20240522-20240528 CAISO Day-Ahead Price','SCE',0) #start the function with the name of the file

        #"BOP.sensorW.W","PowerDemand.y","DNI_Input.y[1]","CosEff_Input.y[1]", "BOP.deaerator.medium.p","dual_Pipe_CTES_Controlled_Feedwater.CTES.T_Ave_Conc","BOP.sensor_T2.T","SMR_Taveprogram.core.Q_total.y" 

        pressures = []

        for u in range(len(results["BOP.deaerator.medium.p"])):
            pressures.append(results["BOP.deaerator.medium.p"][u]/101000)


        conctemps = []

        for c in range(len(results["dual_Pipe_CTES_Controlled_Feedwater.CTES.T_Ave_Conc"])):
            conctemps.append(results["dual_Pipe_CTES_Controlled_Feedwater.CTES.T_Ave_Conc"][c]-273)

        feedtemps = []

        for f in range(len(results["BOP.sensor_T2.T"])):
            feedtemps.append(results["BOP.sensor_T2.T"][f]-273)

        dnis = []

        for d in range(len(results["DNI_Const.y"])):
            dnis.append(results["DNI_Const.y"][d])

        ts = []

        for ti in range(len(results["Time"])):
            ts.append(results["Time"][ti]/ (3600*24))


        fig, axs = plt.subplots(2,3, figsize=(20,12))

        
        fig.suptitle(f"Profit = ${score}")
        axs[0,0].plot(ts,results["PowerDemand.y"], label = 'Demand')
        axs[0,0].set_title('Demands')
        axs[1,0].set(ylabel='Demand / MW')
    
        #axs[0].set_xlim(0,250)
        #axs[0].legend()
        #axs[0].set_title('Temperature')
        # axs[0, 1].plot(t1, mdots, 'tab:orange')
        # axs[0, 1].set_title('Pump Mdot')
        axs[0,0].plot(ts,results["BOP.sensorW.W"], label = 'Power')
        axs[1,0].set_title('Prices')
        axs[1,0].set(ylabel='Price/ $/MWh')
        axs[1,0].plot(times, price_list, 'tab:red')

        axs[1,1].set_title('Average Concrete Temperature')
        axs[1,1].plot(ts, conctemps, 'tab:red')
        axs[1,1].set(ylabel='Temperature / $\circ$ C')

        axs[0,1].set_title('Deaerator Pressure')
        axs[0,1].plot(ts, pressures, 'tab:red')
        axs[0,1].set(ylabel='Pressure / bar')
        axs[0,1].axhspan(1, 1.6, color='green', alpha=0.5, label = "Pressure Band")

        #axs[0,2].set_title('Profit')
        #axs[0,2].plot(tscore, scorelist, 'tab:red')
        #axs[0,2].set(ylabel='Profit / $')

        axs[0,2].set_title('DNI')
        axs[0,2].plot(ts, dnis, 'tab:red')
        axs[0,2].set(ylabel='DNI / MW/m^2')

        axs[1,2].set_title('Feedwater Temperature')
        axs[1,2].plot(ts, feedtemps, 'tab:red')
        axs[1,2].set(ylabel='Temperature / $\circ$ C')
        axs[1,2].axhspan(147.5, 148.5, color='green', alpha=0.5, label = "Pressure Band")

        # axs[1, 1].set_title('TCV pressure drop')
        fig.tight_layout()
        axs[0,0].set(xlabel='Time / days')
        axs[1,0].set(xlabel='Time / days')
        axs[0,1].set(xlabel='Time / days')
        axs[1,1].set(xlabel='Time / days')
        axs[0,2].set(xlabel='Time / days')
        axs[1,2].set(xlabel='Time / days')
            
        #display.display(plt.gcf())
        plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/PPO_Ecom3_1/Iteration_{i}.png")
        plt.close()

        plt.plot(rewards)
        plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/PPO_Ecom3_1/Rewards.png")
        plt.close()
        # Print the final score
        print(f"FINAL SCORE: {score} - Temperature: {0}") # Print the final score
    
    if i % 50 == 0:
        checkpoint_dir = algo.save(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Checkpoints/TrainedPPOEcom3_1_{i}").checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")
        