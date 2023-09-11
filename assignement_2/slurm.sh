#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=test
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=lars.ostberg.moan@gmail.com
#SBATCH -o test_out.txt
#SBATCH -e test_err.txt
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

conda activate comp3710-env

python ./test.py