# Fork Detection: Synthetic Data and additional Experiments

Learning-based fork detection baselines (DINO + MLP, Faster R-CNN) for real and synthetic TEM tiles, plus the synthetic data generator.

| file | |
|---|---|
| `data.py` | HF download, 1–99 % stretch, 0.5× cache, splits, crop sampler |
| `dino.py` | DINOv3/DINOv2 (frozen or last blocks tuned) + MLP per patch token → fork heatmap → points |
| `frcnn.py` | Faster R-CNN R50-FPN v2 on boxes around points → points |
| `evaluate.py` | greedy 75 px matching, class-agnostic / class-dependent P/R/F1 |
| `report.py` | tables (`results/tables`) and figures |
| `synthetic/` | mask generator, TEM renderer, dataset CLI (bit-identical to `Sythetic_TEM_Images`) |
| `slurm/` | sbatch scripts; `runs.txt` lists every run in the tables |
| `notebooks/experiments.ipynb` | rerun, tables, example predictions |

Site settings (python env on `PATH`, `FORKDET_DATA`) go in an untracked `slurm/local.sh`; pass partition/QoS at submission (`sbatch -p <partition> ...`).

```bash
sbatch slurm/data.sbatch
sbatch slurm/dino.sbatch --dataset real --seed 0
sbatch slurm/frcnn.sbatch --dataset synthetic --seed 0
sbatch slurm/queue.sbatch slurm/runs.txt   # all runs behind results/tables, sequentially
sbatch slurm/report.sbatch                 # tables + executed notebook
```
