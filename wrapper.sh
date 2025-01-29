#!/bin/bash
#SBATCH --partition=primary
#SBATCH --job-name=eval7b
#SBATCH --output=/home/vis-t13u/vis-t13u/logs/eval7b_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR1@90

[[ -e env.sh ]] && . env.sh

export OMP_NUM_THREADS=16
export NUMBA_NUM_THREADS=16
export PET_NPROC_PER_NODE=1

srun python eval.py --output_path outputs/llava_onevision_qwen2_7b.txt
