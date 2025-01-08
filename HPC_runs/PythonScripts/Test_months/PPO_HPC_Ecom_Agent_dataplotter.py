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
import pandas as pd

sys.path.append("/home/rigbac/Projects/ALPACA/HPC_runs/PythonScripts")
from matreader import matreader
from PriceReader import importdict, importdict3
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
"""
algo = (
    PPOConfig()
    .env_runners(num_env_runners=80,rollout_fragment_length=1)
    .training(train_batch_size=(320), lr_schedule = [[0,1e-4],[200000,1e-5],[200000,1e-6]])
    .resources(num_cpus_per_worker=0.5)
    .environment(env= "ReactorEcom-v12")
    .build()
)
"""

def write_list_as_row_to_csv(file_name, input_list, t, penalty):
    input_list = np.append(t,input_list)
    input_list = np.append(input_list, penalty)
    # Open the CSV file in append mode
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the input list as a new row
        writer.writerow(input_list)

rewards = []

os.makedirs('/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestMonths/Agent_Paper', exist_ok=True)

for m in [0,5]:

    os.makedirs(f'/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestMonths/Agent_Paper/Month_{m}', exist_ok=True)

    os.chdir('/home/rigbac/Projects/ALPACA')

    os.system('cp -a -f /home/rigbac/Projects/ALPACA/HPC_runs/FileStore_DontDisturb/. /home/rigbac/Projects/ALPACA/HPC_runs/DSFinalStore3')

    env_name = "ReactorEcom-v13"
    env = gym.make(env_name)

    

    for i in [0,4]:

        csv_file = f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestMonths/Agent_Paper/Month_{m}/output_month_agent.csv"

        price_schedule,prices,DNIs,IAMs = importdict3('/home/rigbac/Projects/ALPACA/HPC_runs/Input/Test',m*720 + i*120 ,2022) #start the function with the name of the file

        
        #if i < 100:
        #    algo = Algorithm.from_checkpoint(f"/home/rigbac/ray_results/PPO_2024-09-26_08-22-42/PPO_ReactorEcom-v11_cd677_00001_1_train_batch_size=160_2024-09-26_08-22-46/checkpoint_0000{i}")
        #else:
        #    algo = Algorithm.from_checkpoint(f"/home/rigbac/ray_results/PPO_2024-09-26_08-22-42/PPO_ReactorEcom-v11_cd677_00001_1_train_batch_size=160_2024-09-26_08-22-46/checkpoint_000{i}")
        algo = Algorithm.from_checkpoint("/home/rigbac/ray_results/PPO_2024-09-26_08-22-42/PPO_ReactorEcom-v11_cd677_00001_1_train_batch_size=160_2024-09-26_08-22-46/checkpoint_000029")
        #algo = Algorithm.from_checkpoint("/home/rigbac/Projects/ALPACA/HPC_runs/Output/Checkpoints/TrainedPPOEcom_new_3_2_720")
        
        #result = algo.train()
        #print(pretty_print(result))
        
        observation, info = env.reset(options={'period':i, 'month':m})

        # Reset the score. The final score will be the total amount of steps before the pole falls
        score = 0
        profit = 0
        t = 1003600
        powers = []
        price_list = []
        demands = []
        times = []
        average_power_list=[]
        scorelist = [0]
        tscore = [t]
        t_price = 0

        terminated = False
        truncated = False

        write_list_as_row_to_csv(csv_file, observation, t - 3600 + 3600*120*i, 0)

        for z in range(len(price_schedule)):
            if price_schedule[z] <= t_price < price_schedule[z+1]:
                print(t_price)
                priceforwindow = prices[z]
                print(priceforwindow)

        

        # Go on until the pole falls off
        while not (terminated or truncated):
            print(observation)
            #print(observation[-3])
            action, state, extra = algo.compute_single_action(observation, full_fetch = True, explore = False)

            #action = 3

            ramprate = 2000

            current_power = (observation[0]*1e7)+53.35e6

            print(f'Action = {action}')
            #print(action)
            next_observation, reward, terminated, truncated, results = env.step(action)

            print(next_observation, reward, terminated, truncated)

            final_power = (next_observation[0]*1e7)+53.35e6

            average_power = ((0.5*ramprate*(final_power+current_power)) + (3600-ramprate)*final_power)/3600

            average_power_list.append(average_power)
        


            times.append((t- 1003600)/(3600*24))
            times.append((t+3599 - 1003600)/(3600*24))
            for z in range(len(price_schedule)):
                if price_schedule[z] <= t_price < price_schedule[z+1]:
                    print(t_price)
                    priceforwindow = prices[z]
                    print(priceforwindow)

            price_list.append(priceforwindow)
            price_list.append(priceforwindow)
            score += (reward)*1000

            profit += average_power*(observation[-5]*116.32134)/1e6
            t+=3600
            t_price+=3600

            tscore.append(t/(3600*24))
            scorelist.append(score)
                
            #print(reward)

            # Update the final score (+1 for each step)
            #score += reward

            # Apply penalty for bad state
            if terminated or truncated: # if the pole has fallen down 
                next_observation = None

                    # Write to CSV
            if next_observation is None:
                pass
            else:
                write_list_as_row_to_csv(csv_file, next_observation, t - 3600 + 3600*120*i, float(results['penalty']))
            
            observation = next_observation

        tid = results['tid']

        os.system('cp -a -f /scratch/rigbac/ThreadFilesFWHPT/' + str(tid) + '/dsres.mat /home/rigbac/Projects/ALPACA/HPC_runs/DSFinalStore3')
        os.system('cp -a -f /scratch/rigbac/ThreadFilesFWHPT/' + str(tid) + '/dsfinal.txt /home/rigbac/Projects/ALPACA/HPC_runs/DSFinalStore3')
        
        r1 = random.randint(0, 10000)
        print(results['penalty'])
        
        """
        if not (results['penalty']):
            print('Safe Ending - creating end point')
            try:
                os.mkdir('/home/rigbac/Projects/ALPACA/HPC_runs/Output/StartPoints/' + str(i)+'_'+str(r1))
                os.system('cp -a -f /scratch/rigbac/ThreadFilesFWHPT/' + str(tid) + '/dsres.mat /home/rigbac/Projects/ALPACA/HPC_runs/Output/StartPoints/' + str(i)+'_'+str(r1))
                os.system('cp -a -f /scratch/rigbac/ThreadFilesFWHPT/' + str(tid) + '/dsfinal.txt /home/rigbac/Projects/ALPACA/HPC_runs/Output/StartPoints/' + str(i)+'_'+str(r1))
            except:
                continue
        """
        print(f'Profit of 100 hours is: {profit}')

        print(average_power_list)

        df_price = pd.read_csv('/home/rigbac/Projects/ALPACA/HPC_runs/Input/Test2022.csv')

        rewards.append(score)

        #price_schedule,prices = importdict('/home/rigbac/Projects/ALPACA/HPC_runs/Input/20240522-20240528 CAISO Day-Ahead Price','SCE',0) #start the function with the name of the file

        #"BOP.sensorW.W","PowerDemand.y","DNI_Input.y[1]","CosEff_Input.y[1]", "BOP.deaerator.medium.p","dual_Pipe_CTES_Controlled_Feedwater.CTES.T_Ave_Conc","BOP.sensor_T2.T","SMR_Taveprogram.core.Q_total.y" 

        pressures = []

        for u in range(len(results["BOP.deaerator.medium.p"])):
            pressures.append(results["BOP.deaerator.medium.p"][u]/101000)

        demands_tab = []

        for h in range(len(results["PowerDemand.y"])):
            demands_tab.append(results["PowerDemand.y"][h]/1000000)

        powers_tab = []

        for q in range(len(results["BOP.sensorW.W"])):
            powers_tab.append(results["BOP.sensorW.W"][q]/1000000)


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
            ts.append((results["Time"][ti]-1003600)/ (3600*24))



        fig, axs = plt.subplots(3,2, figsize=(16,11))
        #axs = axs.flatten()
        
        #fig.suptitle(f"Profit = ${score}")
        axs[0,0].plot(ts,demands_tab, label = 'Demand')
        axs[0,0].plot(ts,powers_tab, label = 'Power', linestyle='dashed')
        axs[0,0].set_title('Demands')
        axs[0,0].legend()
        axs[0,0].set(ylabel='Demand / MW', xlabel = 'Time / Days')

        axs[0,0].annotate(
        '(i)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

        #axs[0].set_xlim(0,250)
        #axs[0].legend()
        #axs[0].set_title('Temperature')
        # axs[0, 1].plot(t1, mdots, 'tab:orange')
        # axs[0, 1].set_title('Pump Mdot')
        
        axs[1,0].set_title('Prices')
        axs[1,0].set(ylabel='LMP / $/MWh')
        axs[1,0].plot(times, price_list)

        axs[1,0].annotate(
        '(iii)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

        axs[2,0].set_title('DNI')
        axs[2,0].plot(ts, dnis)
        axs[2,0].set(ylabel='DNI / W/m$^2$')

        axs[2,0].annotate(
        '(v)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

        # axs[1, 1].set_title('TCV pressure drop')
        #fig.tight_layout(h_pad=2)
        axs[1,0].set(xlabel='Time / days')
        axs[2,0].set(xlabel='Time / days')
        axs[0,0].set(xlabel='Time / days')

        #plt.subplots_adjust(bottom=0.05)
        #display.display(plt.gcf())
        #plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestMonths/Agent_Paper/Month_{m}/Iteration_{i}_Demands.png")
        #plt.close()

        #fig, axs = plt.subplots(3,1, figsize=(8,11))
        #axs = axs.flatten()

        axs[2,1].set_title('Average Concrete Temperature')
        axs[2,1].set(ylabel='Temperature / $^\circ$C')
        axs[2,1].axhspan(166.5, 170.5, color='green', alpha=0.4, label = "Desired temperature band")
        axs[2,1].plot(ts, conctemps, 'tab:blue')

        axs[2,1].annotate(
        '(vi)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

        

        axs[1,1].set_title('Deaerator Pressure')
        axs[1,1].set(ylabel='Pressure / bar')
        axs[1,1].axhspan(1.1, 1.6, color='green', alpha=0.4, label = "Desired pressure band")
        axs[1,1].axhspan(1.6, 2, color='orange', alpha=0.4)
        axs[1,1].axhspan(1.0, 1.1, color='orange', alpha=0.4, label = "Penalty zone")
        axs[1,1].set_ylim(0.8, 2.2)
        axs[1,1].plot(ts, pressures, 'tab:blue')
        axs[1,1].legend()

        axs[1,1].annotate(
        '(iv)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

        #axs[0,2].set_title('Profit')
        #axs[0,2].plot(tscore, scorelist, 'tab:red')
        #axs[0,2].set(ylabel='Profit / $')

        axs[0,1].set_title('Feedwater Temperature')
        axs[0,1].set(ylabel='Temperature / $^\circ$C')
        axs[0,1].axhspan(147.5, 148.5, color='green', alpha=0.4, label = "Desired temperature band")
        axs[0,1].axhspan(148.5, 150, color='orange', alpha=0.4)
        axs[0,1].axhspan(146, 147.5, color='orange', alpha=0.4, label = "Penalty zone")
        axs[0,1].plot(ts, feedtemps, 'tab:blue')
        axs[0,1].set_ylim(145.5,150.5)
        axs[0,1].legend()

        axs[0,1].annotate(
        '(ii)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

        # axs[1, 1].set_title('TCV pressure drop')
        axs[1,1].set(xlabel='Time / days')
        axs[2,1].set(xlabel='Time / days')
        axs[0,1].set(xlabel='Time / days')

        fig.tight_layout(h_pad = 2,w_pad=1.5)

        plt.subplots_adjust(bottom=0.05)
        #display.display(plt.gcf())
        plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestMonths/Agent_Paper/Month_{m}/Iteration_{i}_All.png")
        plt.close()

        plt.plot(rewards)
        plt.savefig(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Figures/TestMonths/Agent_Paper/Month_{m}/Rewards.png")
        plt.close()
        # Print the final score
        print(f"FINAL SCORE: {score} - Temperature: {0}") # Print the final score

    #if i % 50 == 0:
        #checkpoint_dir = algo.save(f"/home/rigbac/Projects/ALPACA/HPC_runs/Output/Checkpoints/PPOFinal5_year_{i}").checkpoint.path
        #print(f"Checkpoint saved in directory {checkpoint_dir}")
        