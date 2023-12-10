#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J pythonCPPTest
#SBATCH -o pythonCPPTest.out
#SBATCH -e pythonCPPTest.err
#SBATCH -t 0-12:00:00
#SBATCH -c 1

cd $SLURM_SUBMIT_DIR

module load anaconda/full/2021.05
bootstrap_conda
conda create --name pythonCPPTest
conda activate pythonCPPTest

g++ rk4.cpp -Wall -O3 -std=c++17 -o rk4

python pythonCPPTest.py

