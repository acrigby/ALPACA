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

python SequentionalQlearnCPP.py 1
python SequentionalQlearnCPP.py 2
python SequentionalQlearnCPP.py 3
python SequentionalQlearnCPP.py 4
python SequentionalQlearnCPP.py 5
python SequentionalQlearnCPP.py 6
python SequentionalQlearnCPP.py 7
python SequentionalQlearnCPP.py 8
python SequentionalQlearnCPP.py 9
python SequentionalQlearnCPP.py 10

python ParallelQlearnCPPdequeue.py 1
python ParallelQlearnCPPdequeue.py 2
python ParallelQlearnCPPdequeue.py 3
python ParallelQlearnCPPdequeue.py 4
python ParallelQlearnCPPdequeue.py 5
python ParallelQlearnCPPdequeue.py 6
python ParallelQlearnCPPdequeue.py 7
python ParallelQlearnCPPdequeue.py 8
python ParallelQlearnCPPdequeue.py 9
python ParallelQlearnCPPdequeue.py 10