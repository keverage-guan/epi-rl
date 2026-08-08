#!/bin/bash
#SBATCH --job-name=pinn_fixed_train
#SBATCH --account=def-divya03
#SBATCH --partition=compute
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=22:00:00
#SBATCH --chdir=/scratch/mbarajas/epi-rl
#SBATCH --output=/scratch/mbarajas/epi-rl/logs/%x-%j.out
#SBATCH --error=/scratch/mbarajas/epi-rl/logs/%x-%j.out

set -uo pipefail

cd /scratch/mbarajas/epi-rl

module load python/3.11.5
source .venv/bin/activate
export DDE_BACKEND=pytorch

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m pinn_seir.fit_pinn \
  --census data/great_brittain/census.csv \
  --commute data/great_brittain/commute.csv \
  --crosswalk data/great_brittain/crosswalk.tsv \
  --contacts data/contacts \
  --flu data/epidemic/uk_flu_per_100000.csv \
  --adam-iters 20000 \
  --dropout-rate 0.1 \
  --n-schedules 32 \
  --n-collocation 64 \
  --w-phys 100 \
  --checkpoint-every 5000 \
  --out outputs/seir_pinn/trillium_fixed_run

echo "Job finished: $(date)"
