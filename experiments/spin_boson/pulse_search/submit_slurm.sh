#!/bin/bash
# Slurm job array for the pulse-search experiment on USTC-SCC
# (Intel Xeon 6248 system; cnode = 40 cores / 192 GB, shared up to 4 jobs/node).
#
# One array task per gallery pulse. Keep --array in sync with the gallery:
#     python -m experiments.spin_boson.pulse_search.pulse_gallery --list
# (N pulses -> --array=0-<N-1>)
#
# Submit FROM THE REPOSITORY ROOT (logs need the directory to exist first):
#     mkdir -p logs && sbatch experiments/spin_boson/pulse_search/submit_slurm.sh
# Full guide: experiments/spin_boson/pulse_search/scc_deploy.md
#
#SBATCH -J pulse-search
#SBATCH -o logs/%x_%A_%a.out        # %A = array master id, %a = task id
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p CPU-Little               # single-node tasks -> CPU-Little
#SBATCH --qos=qos_cpu_little        # QoS must match the partition or sbatch rejects with InvalidQOS
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10                       # match runtime.workers in search.yaml
#SBATCH --time=1-00:00:00           # D-HH:MM:SS; CPU-Little allows up to 7 days
#SBATCH --array=0-9

# --- environment -------------------------------------------------------------
source /etc/profile                 # required on compute nodes, else `module: command not found`
set -euo pipefail                   # after /etc/profile, which may reference unset vars
cd "$SLURM_SUBMIT_DIR"
module purge
module load anaconda3/2025.06       # cluster Python (numpy/scipy included)

# Project venv from scc_deploy.md step 2; falls back to the module python.
VENV="${VENV:-$PWD/.venv}"
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

# The driver parallelizes over state pairs with worker processes; keep the
# linear-algebra libraries single-threaded so cores are not oversubscribed.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- run one gallery pulse ----------------------------------------------------
CONFIG="${CONFIG:-experiments/spin_boson/pulse_search/search.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/spin_boson/pulse_search/outputs}"

python -m experiments.spin_boson.pulse_search.run_search \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --index "$SLURM_ARRAY_TASK_ID"
