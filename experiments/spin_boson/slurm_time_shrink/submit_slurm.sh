#!/bin/bash
# Slurm job array for the time-shrink experiment on USTC-SCC
# (Intel Xeon 6248 system; cnode = 40 cores / 192 GB, shared up to 4 jobs/node).
#
# One array task per (pulse, starting time): 10 pulses x {orig, 300us, 170us}
# = 30 tasks. Keep --array in sync with the grid:
#     python -m experiments.spin_boson.slurm_time_shrink.run_time_shrink --list
# (N tasks -> --array=0-<N-1>)
#
# Submit FROM THE REPOSITORY ROOT (logs need the directory to exist first):
#     mkdir -p logs && sbatch experiments/spin_boson/slurm_time_shrink/submit_slurm.sh
# Full guide: experiments/spin_boson/slurm_time_shrink/README.md
#
#SBATCH -J time-shrink
#SBATCH -o logs/%x_%A_%a.out        # %A = array master id, %a = task id
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p CPU-Little               # single-node tasks -> CPU-Little
#SBATCH --qos=qos_cpu_little        # QoS must match the partition or sbatch rejects with InvalidQOS
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10                       # match runtime.workers in shrink.yaml
#SBATCH --time=7-00:00:00           # CPU-Little cap; each task is up to ~30 sequential optimizations
#SBATCH --array=0-29

# --- environment -------------------------------------------------------------
source /etc/profile                 # required on compute nodes, else `module: command not found`
set -euo pipefail                   # after /etc/profile, which may reference unset vars
cd "$SLURM_SUBMIT_DIR"
module purge
module load anaconda3/2025.06       # cluster Python (numpy/scipy included)

# Project venv from the README's setup step; falls back to the module python.
VENV="${VENV:-$PWD/.venv}"
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

# The driver parallelizes over state pairs with worker processes; keep the
# linear-algebra libraries single-threaded so cores are not oversubscribed.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- run one shrink task ------------------------------------------------------
CONFIG="${CONFIG:-experiments/spin_boson/slurm_time_shrink/shrink.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/spin_boson/slurm_time_shrink/outputs}"
SHRINK_FACTOR="${SHRINK_FACTOR:-0.95}"
MAX_ROUNDS="${MAX_ROUNDS:-30}"

python -m experiments.spin_boson.slurm_time_shrink.run_time_shrink \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --shrink-factor "$SHRINK_FACTOR" \
    --max-rounds "$MAX_ROUNDS" \
    --index "$SLURM_ARRAY_TASK_ID"
