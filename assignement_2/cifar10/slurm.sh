#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=LARSERN_test
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=lars.ostberg.moan@gmail.com
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

conda activate comp3710-env

python cifar10_train.py