cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f slurm/local.sh ] && source slurm/local.sh
export FORKDET_DATA=${FORKDET_DATA:-$PWD/data}
export HF_HOME=${HF_HOME:-$FORKDET_DATA/hf_home} TORCH_HOME=${TORCH_HOME:-$FORKDET_DATA/torch_home}
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
