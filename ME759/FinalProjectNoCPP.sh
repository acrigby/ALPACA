#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J finalProjectNoCPP
#SBATCH -o finalProjectNoCPP.out
#SBATCH -e finalProjectNoCPP.err
#SBATCH -t 0-12:00:00
#SBATCH -c 10

cd $SLURM_SUBMIT_DIR

module load anaconda/full/2021.05
bootstrap_conda
conda create --name final_project_nocpp
conda activate final_project_nocpp

conda install pytorch matplotlib IPython

cd ..
pip install ./gymnasium
cd ME759

python SequentionalQlearn.py
python ParallelQlearn.py

