
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:29:29 2023

@author: aidan
"""

import time
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
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process, Manager, Lock, Queue
from multiprocessing.managers import BaseManager
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from SolverClasses import *
import time

SENTINEL = None # no more records sentinel
semaphore = multiprocessing.Semaphore(1)

class DequeManager(BaseManager):
    pass

class DequeProxy(object):
    def __init__(self, *args):
        self.deque = collections.deque(*args)
    def __len__(self):
        return self.deque.__len__()
    def appendleft(self, x):
        self.deque.appendleft(x)
    def append(self, x):
        self.deque.append(x)
    def pop(self):
        return self.deque.pop()
    def popleft(self):
        return self.deque.popleft()

# Currently only exposes a subset of deque's methods.
DequeManager.register('DequeProxy', DequeProxy,
                      exposed=['__len__', 'append', 'appendleft',
                               'pop', 'popleft'])


process_shared_deque = None  # Global only within each process.

def episode_func(episode_num, tau, plotting_rewards, replay_mem, policy_net, target_net, queue):
    print(id(policy_net))
    print(f"Process={os.getpid()}")
    print(episode_num, tau)
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
            next_observation = None

        item = [observation, action, next_observation, reward]

        queue.put(item, block = False)

        #print(replay_mem)
        # Update the replay memory
        semaphore.acquire()

        replay_mem.push(observation, action, next_observation, reward)

        if score % 8 == 1:
            # Update the network
            if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
                  update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size, device)

        semaphore.release()
        observation = next_observation 
    # Print the final score
    print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau}") # Print the final score
    print(len(replay_mem))
    #print(plotting_rewards)
    plotting_rewards.append(score)
    queue.put(SENTINEL)

# consume work
def updater(replay_mem,policy_net, target_net,queue):
    print('Consumer: Running', flush=True)
    # consume work
    while True:
        print(queue.qsize())
        # get a unit of work
        item = queue.get()
        #print(f'>got {item}', flush=True)
        observation = item[0]
        action = item[1]
        next_observation = item[2]
        reward = item[3]
        replay_mem.push(observation, action, next_observation, reward)
        # check for stop
        if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
            update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size, device)
    
    print('Consumer: Done', flush=True)
    return replay_mem,policy_net, target_net
    

def main(plotting_rewards, queue):
    global start_time,target_net_update_steps
    global gamma, optimizer, loss_fn, batch_size, device, min_samples_for_training
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

    global env
    ### Create environment
    env = gym.make('AcrobotCdyn') # Initialize the Gym environment

    # Get the shapes of the state space (observation_space) and action space (action_space)
    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    print(f"STATE SPACE SIZE: {state_space_dim}")
    print(f"ACTION SPACE SIZE: {action_space_dim}")

    ### Initialize the replay memory
    replay_mem = Bmanager.ReplayMemory(replay_memory_capacity) 


    ### Initialize the policy network
    policy_net = DQN(state_space_dim, action_space_dim).to(device)
    policy_net.share_memory()

    ### Initialize the target network with the same weights of the policy network
    target_net = DQN(state_space_dim, action_space_dim).to(device)
    target_net.share_memory()
    target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

    
    ### Initialize the optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr) # The optimizer will update ONLY the parameters of the policy network

    ### Initialize the loss function (Huber loss)
    loss_fn = nn.SmoothL1Loss()

    # Initialize the Gym environment

    env = gym.make('AcrobotCdyn') 
    observation, info = env.reset()
    max_workers=8
    start_time = time.time()

    for episode_collection in range(len(exploration_profile)//max_workers):
        processes = []
        seen_sentinel_count = 0

        for episode in range(max_workers):

            # Launch the first round of tasks, building a list of ApplyResult objects
            process = multiprocessing.Process(target = episode_func,args = (episode_collection*max_workers + episode, exploration_profile[episode_collection*max_workers + episode], plotting_rewards, replay_mem,policy_net, target_net, queue))
            processes.append(process)
            
        for p in processes:
            p.start()

        for p in processes:
            p.join()


        while seen_sentinel_count < max_workers:
            #print(queue.qsize())
            # get a unit of work
            item = queue.get(block = False)
            #print(f'>got {item}', flush=True)
            if item is SENTINEL:
                seen_sentinel_count += 1
            elif item is None:
                seen_sentinel_count += 1
            else:
                #print(item[0])
                observation = item[0]
                action = item[1]
                next_observation = item[2]
                reward = item[3]
                replay_mem.push(observation, action, next_observation, reward)
                # check for stop

        # report
        # all done
        print('Consumer: Done', flush=True)
        print("main replay mem")
        print(len(replay_mem))
        print("main policy net")
        print(policy_net)
        print('Updating target network main')
        target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network
        if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
            update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size, device)
        print("updating outside")
        time.sleep(0.5)

if __name__ == "__main__":
    with Manager() as manager:
        plotting_rewards = manager.list([])
        queue = manager.Queue() 

        BaseManager.register('ReplayMemory', ReplayMemory)
        Bmanager = BaseManager()
        Bmanager.start()
        main(plotting_rewards, queue)
        fig = plt.figure()
        ax = fig.add_subplot()

        fig.suptitle('Parallel Plotting Rewards CPP', fontsize=10, fontweight='bold')
        ax.set_title("--- %s seconds ---" % (time.time() - start_time))
        ax.plot(plotting_rewards)
        plt.savefig('ParCPP.png')

        print("--- %s seconds ---" % (time.time() - start_time))