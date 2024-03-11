#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J pythonTestRun
#SBATCH -o pythonTestRun.out
#SBATCH -e pythonTestRun.err
#SBATCH -t 0-00:30:00
#SBATCH -c 1

cd $SLURM_SUBMIT_DIR

module load anaconda/full/2021.05
bootstrap_conda
conda install numpy matplotlib
conda create --name test_env numpy matplotlib
conda activate test_env

python pyTest.py