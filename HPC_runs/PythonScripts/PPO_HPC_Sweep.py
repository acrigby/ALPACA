# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 08:09:44 2024

@author: aidan
"""

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
from pathlib import Path

# Note: `gymnasium` (not `gym`) will be **the** API supported by RLlib from Ray 2.3 on.
import gymnasium as gym

import os
import sys
import torch
import json
import numpy as np
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch import nn
from csv import writer
import os
import logging, shutil, pickle, pprint

norm = matplotlib.colors.Normalize(vmin=9, vmax=15)
cmap = matplotlib.colormaps['rainbow']

from time import process_time

sys.path.append("/home/rigbac/Projects/ALPACA/HPC_runs/PythonScripts")
from matreader import matreader

print(os.path.abspath(os.curdir))
#os.chdir("..")
print(os.path.abspath(os.curdir))
ALPACApath = os.path.abspath(os.curdir)

f = open("./Utilities/ALPACAASCII.txt", 'r')
print(''.join([line for line in f]))

#register new enviroments
#exec(open(str(ALPACApath)+"/gymnasium/gymnasium/envs/classic_control/__init__.py").read())

env_name = "Reactor-v8"
env = gym.make(env_name)

os.chdir('/home/rigbac/Projects/ALPACA')

# set up matplotlib
#is_ipython = 'inline' in matplotlib.get_backend()
#if is_ipython:
#    from IPython import display

#plt.ion()

#define orginial results profile for plotting
Orig_variables = ["Time","sensor_pT.T"]    
Orig_results = matreader(str(ALPACApath)+"/HPC_runs/DymolaTemplates/Original_Temp_Profile.mat", Orig_variables)
    
for i in range(0,len(Orig_results["Time"]),1):
    Orig_results["Time"][i] = Orig_results["Time"][i] - 9960
    Orig_results["sensor_pT.T"][i] = Orig_results["sensor_pT.T"][i] - 273.15

plt.plot(Orig_results["Time"],Orig_results["sensor_pT.T"], label = 'Original')
plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/FinalPPO/original.png")
plt.close()

"""
algo = (
    SACConfig()
    .env_runners(num_env_runners=80, rollout_fragment_length=1)
    .training(train_batch_size=(2048),lr_schedule=([0,1e-4],[200000,1e-5],[400000,1e-6]))
    .resources(num_cpus_per_worker=0.5,num_gpus=0)
    .environment(env= "Reactor-v8")
    .build()
)"""
"""
hyperparam_mutations = {
    "lambda": lambda: random.uniform(0.9, 1.0),
    "clip_param": lambda: random.uniform(0.01, 0.5),
    "lr": [5e-4, 1e-4, 5e-5, 1e-5, 5e-6],
    "train_batch_size": [512,1024,2048],
}

print("Best performing trial's final set of hyperparameters:\n")
pprint.pprint(
    {k: v for k, v in algo.config.items() if k in hyperparam_mutations}
)
"""



"""
policy = Policy.from_checkpoint('/home/rigbac/ray_results/PPO_2024-06-11_08-14-46/PPO_Reactor-v8_fd033_00001_1_train_batch_size=512_2024-06-11_08-15-00/checkpoint_000415')
policy = policy['default_policy']
weights = policy.get_weights()
weights = {'default_policy': weights}

algo_config = (
    PPOConfig()
    .env_runners(num_env_runners=40, rollout_fragment_length=1)
    .training(train_batch_size=(2048),lr_schedule=[[0,1e-5], [100000, 1e-7]])
    .resources(num_cpus_per_worker=1,num_gpus=4)
    .environment(env= "Reactor-v8")
)

print(algo_old.config.items())

algo_config.update_from_dict(dict(algo_old.config.items()))

algo_config.update_from_dict({"lr_schedule":[[0,1e-5], [100000, 1e-7]], "num_env_runners":40, "num_cpus_per_worker":1, "num_gpus":4})

algo = algo_config.build()

algo.set_weights(weights)

pprint.pprint(
    {k: v for k, v in algo.config.items() if k in ["CPUS", "lr"]}
)

#algo.config.env_runners(num_env_runners=PPOConfig.overrides(num_env_runners = 40))

"""

rewards_dict = {}
rewards = []
ramprates = []
ramps_dict = {}
means = {}

for j in range(1020,1021,1):

    results_list_j = []

    #algo = Algorithm.from_checkpoint(f'/home/rigbac/Projects/ALPACA/RunData/PPO_2024-06-11_08-14-46/PPO_Reactor-v8_fd033_00004_4_train_batch_size=1024_2024-06-11_08-15-00/checkpoint_000{j}')
    algo = Algorithm.from_checkpoint('/home/rigbac/ray_results/PPO_2024-08-15_16-51-53/PPO_Reactor-v8_fe178_00004_4_train_batch_size=1024_2024-08-15_16-52-05/checkpoint_000167')
    #algo = Algorithm.from_checkpoint('HPC_runs/Output/Figures/PaperTests/DQN/FinalControlImpala_1500')
    #algo = Algorithm.from_checkpoint('/home/rigbac/Projects/ALPACA/RunData/PPO_2024-06-11_08-14-46/PPO_Reactor-v8_fd033_00004_4_train_batch_size=1024_2024-06-11_08-15-00/checkpoint_000678')
    newfolder = f'/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/PaperRayTest/Checkpoint_{j}'
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)
    fig, axs = plt.subplots(2,1)
    axs[0].axhspan(398, 402, alpha=0.5, color='green')
    for i in range(1):
            
        observation, info = env.reset()
    
        # Reset the score. The final score will be the total amount of steps before the pole falls
        score = 0
        t = 0
        temps = []
        mdots = []
        powers = []
        TCVdps = []
        e_times = []
        feeds = []
        t1 = []
        t2= []
        terminated = False
        truncated = False
        
        temps.append(observation[0]+400)
        feeds.append(observation[3])
        t1.append(t*5)
        t2.append(t*5)
        
    
        # Go on until the pole falls off
        while not (terminated or truncated):
            t2.append(t*5)
            t+=1

            t1_start = process_time() 
            action = algo.compute_single_action(observation, explore=False)
            t1_stop = process_time()

            e_time = t1_stop-t1_start

            print(e_time)
            e_times.append(e_time)

            next_observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            
            temps.append(next_observation[0]+400)
            feeds.append(next_observation[3])
            feeds.append(next_observation[3])
            t1.append(t*5)
            t2.append(t*5)
                
            #print(reward)

            # Update the final score (+1 for each step)
            #score += reward

            # Apply penalty for bad state
            if terminated or truncated: # if the pole has fallen down 
                next_observation = None
                height = 10000000 - (observation[2]*1000000)
            
            observation = next_observation
        """
        observation_o, info_o = env.reset()
    
        # Reset the score. The final score will be the total amount of steps before the pole falls
        score_o = 0
        t_o = 0
        temps_o = []
        mdots_o = []
        powers_o = []
        TCVdps_o = []
        feeds_o = []
        t1_o = []
        t2_o= []
        terminated_o = False
        truncated_o = False
        
        temps_o.append(observation_o[0]+673.15)
        feeds_o.append(observation_o[3])
        t1_o.append(t_o*5)
        t2_o.append(t_o*5)
        
    
        # Go on until the pole falls off
        while not (terminated_o or truncated_o):
            t2_o.append(t_o*5)
            t_o+=1

            action_o = [0]
            next_observation_o, reward_o, terminated_o, truncated_o, info_o = env.step(action_o)
            score_o += reward_o
            
            temps_o.append(next_observation_o[0]+673.15)
            feeds_o.append(next_observation_o[3])
            feeds_o.append(next_observation_o[3])
            t1_o.append(t_o*5)
            t2_o.append(t_o*5)
                
            #print(reward)

            # Update the final score (+1 for each step)
            #score += reward

            # Apply penalty for bad state
            if terminated_o or truncated_o: # if the pole has fallen down 
                next_observation_o = None
                height_o = 10000000 - (observation_o[2]*1000000)
            
            observation_o = next_observation_o
            """
        results_list_j.append(score)
        print(e_times)
        print(f'mean elapased CPU_time = {np.average(e_times)}')
        ramprate = (height/400000)/1.666
        ramprate = round(ramprate, 2)
        scoreprint = round(score, 2)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        print(m.to_rgba(ramprate))
        #fig.suptitle(f"Response for a transient ramp rate of {ramprate} %/min -  Episode Reward = {scoreprint}")
        axs[0].plot(t1, temps, color=m.to_rgba(ramprate))
        axs[0].set_xlim(0,250)
        #axs[0].set_ylim(396,408)
        axs[0].set_ylabel('Temperature / $^\circ$C')
        # axs[0, 1].plot(t1, mdots, 'tab:orange')
        # axs[0, 1].set_title('Pump Mdot')
        axs[1].plot(t2, feeds, color=m.to_rgba(ramprate))
        axs[1].set_ylabel('Feedforward Signal / kg/s')
        axs[1].set_xlim(0,250)
        # axs[1, 1].plot(t1, TCVdps, 'tab:red')

        # axs[1, 1].set_title('TCV pressure drop')
        axs[1].set(xlabel='Time / s')
            
    #display.display(plt.gcf())
    axs[0].plot(Orig_results["Time"],Orig_results["sensor_pT.T"], label = 'Original', color=m.to_rgba(15))
    axs[0].legend()
    fig.colorbar(m, ax=axs, label = 'Ramp Rate %/min')
    plt.show()
    plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/PaperRayTest/Checkpoint_{j}/Iteration_{i}.png")

    plt.plot(rewards)
    plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/PaperRayTest/Checkpoint_{j}/Rewards.png")
    plt.close()
    # Print the final score
    print(f"FINAL SCORE: {score} - Temperature: {0}") # Print the final score

    rewards.append(score)
    ramprates.append(ramprate)

    print(rewards)
    print(ramprates)

    rewards_dict[j] = results_list_j
    ramps_dict[j] = ramprates
    mean = np.average(results_list_j)
    means[j] = [ramprate , mean]

    file_dict = {j : results_list_j}
    ramps_dict = {j : ramprates}
    mean_dict = {j : mean}


    with open('/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/PaperRayTest/rewards_dict.txt', 'a') as convert_file: 
        convert_file.write(json.dumps(file_dict))
        convert_file.write('\n')
    with open('/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/PaperRayTest/ramps_dict.txt', 'a') as convert_file: 
        convert_file.write(json.dumps(ramps_dict))
        convert_file.write('\n')
    with open('/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/PaperRayTest/means.txt', 'a') as convert_file: 
        convert_file.write(json.dumps(mean_dict))
        convert_file.write('\n')

print(rewards)
print(ramprates)

with open('/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/PaperRayTest/final_rewards_dict.txt', 'a') as convert_file: 
    convert_file.write(json.dumps(rewards_dict))
with open('/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/PaperRayTest/final_means.txt', 'a') as convert_file: 
    convert_file.write(json.dumps(means))