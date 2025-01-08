# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 08:09:44 2024

@author: aidan
"""

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac.sac import SACConfig
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

import random

import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing"
)
args, _ = parser.parse_known_args()

# Postprocess the perturbed config to ensure it's still valid
def explore(config):
    return config

sys.path.append("/home/rigbac/Projects/ALPACA/HPC_runs/PythonScripts")
from matreader import matreader

print(os.path.abspath(os.curdir))
#os.chdir("..")
print(os.path.abspath(os.curdir))
ALPACApath = os.path.abspath(os.curdir)

f = open("./Utilities/ALPACAASCII.txt", 'r')
print(''.join([line for line in f]))

hyperparam_mutations = {
    "lambda": lambda: random.uniform(0.9, 1.0),
    "clip_param": lambda: random.uniform(0.01, 0.5),
    "lr": [1e-4,5e-5,1e-5,5e-6, 1e-6],
    "train_batch_size": [160,320,640,1280],
}

pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    perturbation_interval=120,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations=hyperparam_mutations,
)

# Stop when we've either reached 100 training iterations or reward=300
stopping_criteria = {"training_iteration": 2000, "episode_reward_mean": 700}

tuner = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        metric="episode_reward_mean",
        mode="max",
        scheduler=pbt,
        num_samples=1 if args.smoke_test else 6,
    ),
    param_space={
        "env": "ReactorEcom-v11",
        "kl_coeff": 1.0,
        "num_workers": 12,
        'num_cpus_per_worker':0.5,
        "num_gpus": 0,  # number of GPUs to use per trial
        # These params are tuned from a fixed starting value.
        "lambda": 0.95,
        "clip_param": 0.2,
        "lr": 1e-4,
        # These params start off randomly drawn from a set.
        "train_batch_size": tune.choice([160,320,640,1280]),
        "rollout_fragment_length":1,
    },
    run_config=train.RunConfig(stop=stopping_criteria),
)
results = tuner.fit()

import pprint

best_result = results.get_best_result()

print("Best performing trial's final set of hyperparameters:\n")
pprint.pprint(
    {k: v for k, v in best_result.config.items() if k in hyperparam_mutations}
)

print("\nBest performing trial's final reported metrics:\n")

metrics_to_print = [
    "episode_reward_mean",
    "episode_reward_max",
    "episode_reward_min",
    "episode_len_mean",
]
pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})