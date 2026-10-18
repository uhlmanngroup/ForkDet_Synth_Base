#!/bin/bash
#SBATCH --job-name=forkdet-report
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
source slurm/env.sh
for f in results/*/preds.csv; do [ -e "$f" ] && gzip -f "$f"; done
python report.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/experiments.ipynb
