#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J finalProjectCPP
#SBATCH -o finalProjectCPP.out
#SBATCH -e finalProjectCPP.err
#SBATCH -t 0-12:00:00
#SBATCH -c 10

cd $SLURM_SUBMIT_DIR

cd ..
git submodule update --init
cd ME759

cd ..
cd ..
python -m venv env-01
source env-01/bin/activate
cd ALPACA

pip install ./gymnasium
cd ME759

pip install torch matplotlib IPython

g++ rk4.cpp -o rk4

python SequentionalQlearnCPP.py 6

python ParallelQlearnCPPdequeue.py 6