#!/bin/bash
#SBATCH --job-name=forkdet-frcnn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
source slurm/env.sh
python frcnn.py "$@"
