# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 07:17:00 2023

@author: localuser
"""

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from DNN_Classes import ReplayMemory, DQN, Transition
from dymola.dymola_interface import DymolaInterface


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    print(eps_threshold)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            print(policy_net(state).max(1)[1].view(1, 1))
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def plot_scores(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_scores, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
def optimize_model():
    
    if len(memory) < BATCH_SIZE:
        print('Too little memory')
        print(len(memory))
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

model = "ControlTests.SteamTurbine_L2_OpenFeedHeat_Test2"

env = gym.make("Reactor-v2")

dymola2 = DymolaInterface()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.005
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

episode_scores = []

steps_done = 0
    
if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50000
    
Orig_results = {}
Orig_variables = ["Time","sensor_pT.T"]    
for key in Orig_variables:
    Orig_results[key] = []
Orig_trajsize = dymola2.readTrajectorySize("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts/Original_Temp_Profile.mat")
Orig_signals = dymola2.readTrajectory("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts/Original_Temp_Profile.mat", Orig_variables, Orig_trajsize)

for i in range(0,len(Orig_variables),1):
    Orig_results[Orig_variables[i]].extend(Orig_signals[i])
    
for i in range(0,len(Orig_results["Time"]),1):
    Orig_results["Time"][i] = Orig_results["Time"][i] - 9900

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    print('##################################################################################')
    print('##################################################################################')
    state, info = env.reset()


    score = 0
    temps = []
    mdots = []
    powers = []
    TCVdps = []
    feeds = []
    t1 = []
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        
        print(reward)
        print(action)
        score += reward

        if terminated:
            next_state = None
            temps.append(observation[0])
            feeds.append(observation[3])
            #mdots.append(observation[4])
            powers.append(observation[2])
            #TCVdps.append(observation[3])
            t1.append(t*5 + 5)
        else:
            temps.append(observation[0])
            feeds.append(observation[3])
            #mdots.append(observation[4])
            powers.append(observation[2])
            #TCVdps.append(observation[3])
            t1.append(t*5 + 5)
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_scores.append(score)
            plot_scores()
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
                
            # plt.plot(t1,temps, label = 'Temperature')
            # plt.axhspan(672.15, 674.15, color='red', alpha=0.5, label = "Temperature Band")
            # plt.legend()
            # display.display(plt.gcf())
            # plt.close()
            # display.clear_output(wait=True)
            # plt.plot(t1,mdots, label = 'Mass Flow Rate')
            # plt.legend()
            # plt.show()
            # display.display(plt.gcf())
            break

print('Complete')
plot_scores(show_result=True)
plt.ioff()
plt.show()

