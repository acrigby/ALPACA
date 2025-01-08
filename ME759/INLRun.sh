#!/bin/bash
#PBS -N test_run_1
#PBS -l select=1:ncpus=1:mem=10gb
#PBS -l walltime=5:00
#PBS -j oe
module load dymola
module load qt
#module load gcc
#module load glib

pwd
cd Projects/ALPACA
source env_RL_2/bin/activate
export PYTHONPATH=/home/rigbac/Desktop/dymola

python ME759/SACRLLib.py
