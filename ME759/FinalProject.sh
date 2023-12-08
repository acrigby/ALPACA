#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J finalProject
#SBATCH -o finalProject.out
#SBATCH -e finalProject.err
#SBATCH -t 0-12:00:00
#SBATCH -c 1

cd $SLURM_SUBMIT_DIR

module load anaconda/full/2021.05
bootstrap_conda
conda create --name final_project
conda activate final_project

conda install pytorch matplotlib

cd ..
pip install gymnasium
cd ME759

python SequentionalQlearnCPP.py
python ParallelQlearnCPP.py

