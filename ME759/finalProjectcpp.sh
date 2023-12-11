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

python -m venv env-01
source env-01/bin/activate

cd ..
pip install ./gymnasium
cd ME759

pip install torch matplotlib IPython

g++ rk4.cpp -o rk4


python delVar.py
python del replay_mem
python del policy_net
python del target_net
python del shared_deque
python ParallelQlearnCPPdequeue.py
