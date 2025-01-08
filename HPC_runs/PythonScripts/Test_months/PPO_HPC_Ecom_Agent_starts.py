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
import csv
import os
import random
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

#env_name = "ReactorEcom-v17"
#env = gym.make(env_name)

os.chdir('/home/rigbac/Projects/ALPACA')

#os.system('cp -a -f /home/rigbac/Projects/ALPACA/HPC_runs/FileStore_DontDisturb/. /home/rigbac/Projects/ALPACA/HPC_runs/DSFinalStore3')


def getState(response):

    #print(reward_price_t)
    
    #rint(reward_price)
    #Set inital state from orginal results "BOP.sensorW.W","PowerDemand.y","DNI_Input.y[1]","CosEff_Input.y[1]", "BOP.deaerator.medium.p","dual_Pipe_CTES_Controlled_Feedwater.CTES.T_Ave_Conc","BOP.sensor_T2.T","SMR_Taveprogram.core.Q_total.y"
    #Power = response["BOP.sensorW.W"][-1]
    PowerDemand = response["PowerDemand.y"][-1]
    #dni = response["DNI_Input.y[1]"][-1]
    #CosEFF = response["CosEff_Input.y[1]"][-1]
    Deaer_P = response["BOP.deaerator.medium.p"][-1]
    Conc_Temp = response["dual_Pipe_CTES_Controlled_Feedwater.CTES.T_Ave_Conc"][-1]
    fwit = response["BOP.sensor_T2.T"][-1]
    #CorePower = response["SMR_Taveprogram.core.Q_total.y"][-1]

    
    state = [PowerDemand, Deaer_P, Conc_Temp, fwit]

    #print(state)

    return state


def write_list_as_row_to_csv(file_name, input_list, t, penalty):
    input_list = np.append(t,input_list)
    input_list = np.append(input_list, penalty)
    # Open the CSV file in append mode
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the input list as a new row
        writer.writerow(input_list)

rewards = []

#os.makedirs('/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestMonths/Agent', exist_ok=True)

startfolders = [ f.path for f in os.scandir('/home/rigbac/Projects/ALPACA/HPC_runs/Output/StartPoints') if f.is_dir() ]

startfolders.insert(0,'/home/rigbac/Projects/ALPACA/HPC_runs/FileStore_DontDisturb')

print(startfolders)

i = 0

for start_folder in startfolders:

    print(i)

    variables = ["Time","BOP.sensorW.W","PowerDemand.y","DNI_Const.y","CosEff_Const.y", "BOP.deaerator.medium.p","dual_Pipe_CTES_Controlled_Feedwater.CTES.T_Ave_Conc","BOP.sensor_T2.T","SMR_Taveprogram.core.Q_total.y" ]
        

    matSourceFileName = start_folder + "/dsres.mat"

    response = matreader(matSourceFileName,variables)

    yout = getState(response)
    
    state = yout

    if yout[0] < 52.5e6:
        if  441.3 < yout[2] < 441.7:
            print(start_folder)
            print(f"start = {yout}")

    i += 1


 