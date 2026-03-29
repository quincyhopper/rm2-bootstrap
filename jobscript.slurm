#!/bin/bash --login
#SBATCH -J train
#SBATCH -p gpuA40GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -t 0-1

module purge

uv run run.py