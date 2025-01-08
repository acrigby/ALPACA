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
from ray.rllib.policy.policy import Policy
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
import logging, pprint

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

#env_name = "ReactorEcom-v13"
#env = gym.make(env_name)

os.chdir('/home/rigbac/Projects/ALPACA')
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






#print(algo_old.config.items())

#algo_config.update_from_dict(dict(algo_old.config.items()))

#algo_config.update_from_dict({"train_batch_size":1000})

algo = Algorithm.from_checkpoint("/home/rigbac/ray_results/PPO_2024-09-26_08-22-42/PPO_ReactorEcom-v11_cd677_00001_1_train_batch_size=160_2024-09-26_08-22-46/checkpoint_000029")

pprint.pprint(
    {k: v for k, v in algo.config.items()}
)

