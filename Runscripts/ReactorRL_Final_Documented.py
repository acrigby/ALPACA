# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:29:29 2023

@author: aidan
"""

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

#name enviroment
enviroment = "Reactor-v3"

#switch off duplicate process checks to prevent errors
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
     
### Define exploration profile
initial_value = 5
num_iterations = 500
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
target_net_update_steps = 5   # Number of episodes to wait before updating the target network
batch_size = 256   # Number of samples to take from the replay memory for each update
bad_state_penalty = 0   # Penalty to the reward when we are in a bad state (in this case when the pole falls down) 
min_samples_for_training = 500   # Minimum samples in the replay memory to enable the training


### Create environment
model = "ControlTests.SteamTurbine_L2_OpenFeedHeat_Test2"
env = gym.make(enviroment) # Initialize the Gym environment
dymola2 = DymolaInterface()


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

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
env = gym.make(enviroment, render_mode = None) 
observation, info = env.reset()

#create arrays to record rewards and scores
plotting_rewards=[]
episode_scores = []

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
    

    
""" Main Loop
        1. Create CSV to store feedforward signals 
        2. reset and run episode
        3. If terminated or truncated - update and plot results
"""
#open csv to store feedforward results
with open('RunData/feeds.csv', 'w', newline='') as f_object:
    writer_object = writer(f_object)
    #loop over expoloration profile to learn
    for episode_num, tau in enumerate(tqdm(exploration_profile)):
    
        # Reset the environment and get the initial state
        observation, info = env.reset()
        # Reset the score. The final score will be the total amount of steps before the pole falls
        score = 0
        t = 0
        #create holding arrays for plotting
        temps = []
        mdots = []
        powers = []
        TCVdps = []
        feeds = []
        writefeeds = []
        t1 = []
        terminated = False
        truncated = False
        
        temps.append(observation[0]+673.15)
        feeds.append(observation[3])
        t1.append(t*5)
    
        # Go on until the pole falls off
        while not (terminated or truncated):
          t+=1
    
          # Choose the action following the policy
          action, q_values = choose_action_softmax(policy_net, observation, temperature=tau)
          
          #print(action, observation)
          
          # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
          next_observation, reward, terminated, truncated, info = env.step(action)
          
          temps.append(next_observation[0]+673.15)
          feeds.append(next_observation[3])
          t1.append(t*5)
             
          print(reward)
    
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
        
        #plot outputs of an episode
        plotting_rewards.append(score)
        episode_scores.append(score)
        plot_scores(episode_scores)
        fig, axs = plt.subplots(2,1)
        axs[0].plot(Orig_results["Time"],Orig_results["sensor_pT.T"], label = 'Original')
        axs[0].plot(t1, temps, label = 'Optimised')
        axs[0].set_xlim(0,max(t1))
        axs[0].axhspan(666.15, 680.15, color='red', alpha=0.35)
        axs[0].axhspan(671.15, 675.15, color='green', alpha=0.5)
        #axs.set_xlim(0,200)
        axs[0].legend(ncol=2)
        axs[0].set(ylabel = 'Temperature / K')
        axs[0].set_title('Temperature')
        # axs[0, 1].plot(t1, mdots, 'tab:orange')
        # axs[0, 1].set_title('Pump Mdot')
        axs[1].plot(t1, feeds, 'tab:green')
        axs[1].set_xlim(0,max(t1))
        axs[1].set_title('FF Signal')
        #yticks = np.arange(-2, 2, 0.5)
        #axs[1].set_yticks(yticks)
        # axs[1, 1].plot(t1, TCVdps, 'tab:red')
        # axs[1, 1].set_title('TCV pressure drop')
        fig.tight_layout()
        axs[1].set(xlabel='Time / s')
            
        display.display(plt.gcf())
        plt.close()
        torch.save(policy_net, 'RunData/DQNOutputpk.pt')
        # Print the final score
        print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau}") # Print the final score
        
        #write feedforward signal to CSV
        writefeeds = feeds
        writefeeds.insert(0,episode_num)
        writer_object.writerow(writefeeds)
    env.close()
    
    plt.plot(plotting_rewards)
    plt.show()
    
    # Initialize the Gym environment
    env = gym.make(enviroment) 
    observation, info = env.reset()
    plotting_rewards_final = []
    
    """Final Loop
    Run with 0 temperature to select best policy based on network
    """
    for episode_num in range(1):
    
        # Reset the environment and get the initial state
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
        t1.append(t*3)
    
        # Go on until the pole falls off
        while not (terminated or truncated):
          t+=1
    
          # Choose the action following the policy
          action, q_values = choose_action_softmax(policy_net, observation, temperature=0)
          
          #print(action, observation)
          
          # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
          next_observation, reward, terminated, truncated, info = env.step(action)
          
          temps.append(next_observation[0]+673.15)
          feeds.append(next_observation[3])
          t1.append(t*5)
             
          #print(reward)
    
          # Update the final score (+1 for each step)
          score += reward
    
          # Apply penalty for bad state
          if terminated or truncated: # if the pole has fallen down 
              reward += bad_state_penalty
              next_observation = None
          
          observation = next_observation
        
        plotting_rewards_final.append(score)
        episode_scores.append(score)
        plot_scores(episode_scores)
        fig, axs = plt.subplots(2,1)
        axs[0].plot(Orig_results["Time"],Orig_results["sensor_pT.T"], label = 'Original')
        axs[0].plot(t1, temps)
        axs[0].axhspan(671.15, 675.15, color='green', alpha=0.5, label = "Temperature Band")
        #axs.set_xlim(0,200)
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
        print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {0}") # Print the final score
    
    env.close()
    
    plt.plot(plotting_rewards_final)
    plt.show()
    f_object.close()