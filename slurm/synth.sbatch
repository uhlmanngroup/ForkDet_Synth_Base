#!/bin/bash
#SBATCH --job-name=forkdet-synth
#SBATCH --array=0-15
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
source slurm/env.sh
python -m synthetic.generate --out "${OUT:-$FORKDET_DATA/synthetic_regen}" --split "${SPLIT:-train}" --shards "${SLURM_ARRAY_TASK_COUNT:-1}" ${COUNT:+--count $COUNT}
