#!/bin/bash
#SBATCH --job-name=forkdet-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
source slurm/env.sh
while read -r line; do [ -n "$line" ] && eval "python $line"; done < "$1"
