#!/bin/bash
#PBS -N test_run_S
#PBS -l select=1:ncpus=1:mem=10gb
#PBS -l walltime=15:00:00
#PBS -j oe
#PBS -e /home/rigbac/Projects/ALPACA/HPC_runs/Output/error_DQN.txt
#PBS -o /home/rigbac/Projects/ALPACA/HPC_runs/Output/output_DQN.txt

module load dymola

pwd
cd Projects/ALPACA
source env_RL_2/bin/activate

python HPC_runs/PythonScripts/DQN_HPC.py