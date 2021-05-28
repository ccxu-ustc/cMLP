#!/bin/bash

#SBATCH -A test
#SBATCH -J cMLP
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -o 100.o
#SBATCH -t 0-12:00:00       
#SBATCH -N 1


module load gcc/5.5.0

python3 cMLP.py --event 100

