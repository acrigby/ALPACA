#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J finalProjectCPP
#SBATCH -o finalProjectCPP.out
#SBATCH -e finalProjectCPP.err
#SBATCH -t 0-12:00:00
#SBATCH -c 5

cd $SLURM_SUBMIT_DIR

module load anaconda/full/2021.05
bootstrap_conda
conda create --name final_project_cpp
conda activate final_project_cpp

conda install pytorch matplotlib IPython

cd ..
pip install ./gymnasium
cd ME759

g++ rk4.cpp -Wall -O3 -std=c++17 -o rk4

python SequentionalQlearnCPP.py

