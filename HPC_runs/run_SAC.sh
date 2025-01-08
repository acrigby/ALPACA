#!/bin/bash
#PBS -N test_run_SAC
#PBS -l select=1:ncpus=1:mem=10gb
#PBS -l walltime=30:00:00
#PBS -j oe
#PBS -e /home/rigbac/Projects/ALPACA/HPC_runs/Output/error_SAC2.txt
#PBS -o /home/rigbac/Projects/ALPACA/HPC_runs/Output/output_SAC2.txt

module load dymola

pwd
cd Projects/ALPACA
source env_RL_2/bin/activate

python HPC_runs/PythonScripts/SAC_HPC.py