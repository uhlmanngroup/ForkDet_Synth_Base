#!/bin/bash
#SBATCH --job-name=forkdet-data
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
source slurm/env.sh
python data.py
