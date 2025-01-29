#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=primary
#SBATCH -J eval_05b
#SBATCH --output=logs/%x_%j_%a.txt
#SBATCH -a 0-32

source venv/bin/activate

srun accelerate launch \
        --main_process_port $((10000 + $RANDOM % 65535)) \
        --num_processes 2 \
        eval_05b.py --output_path outputs/05b_$SLURM_ARRAY_TASK_ID.csv
