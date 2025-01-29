#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=primary
#SBATCH -J eval
#SBATCH --output=logs/%x_%j_%a.txt
#SBATCH -a 0

source venv/bin/activate

srun accelerate launch \
        --main_process_port $((10000 + $RANDOM % 65535)) \
        --num_processes 4 \
        eval.py --output_path outputs/beam/$SLURM_ARRAY_TASK_ID.csv --prompt_type "cot" --decoding "beam"

