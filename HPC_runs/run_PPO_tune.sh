#!/bin/bash
#PBS -N test_run_tune
#PBS -l select=1:ncpus=1:mem=10gb
#PBS -l walltime=15:00:00
#PBS -j oe
#PBS -e /home/rigbac/Projects/ALPACA/HPC_runs/Output/error_SAC.txt
#PBS -o /home/rigbac/Projects/ALPACA/HPC_runs/Output/output_SAC.txt

module load dymola

pwd
#cd Projects/ALPACA
source env_RL_2/bin/activate

python HPC_runs/PythonScripts/SAC_HPC_tune.py