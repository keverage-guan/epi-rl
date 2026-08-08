#!/bin/bash
#SBATCH --job-name=pinn_size_test
#SBATCH --account=def-divya03
#SBATCH --partition=debug
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=00:30:00
#SBATCH --chdir=/scratch/mbarajas/epi-rl
#SBATCH --output=/scratch/mbarajas/epi-rl/logs/%x-%j.out
#SBATCH --error=/scratch/mbarajas/epi-rl/logs/%x-%j.out

set -euo pipefail
cd /scratch/mbarajas/epi-rl
module load python/3.11.5
source .venv/bin/activate
export DDE_BACKEND=pytorch

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for CFG in "16 32" "32 64" "64 128" "96 192"; do
  set -- $CFG
  NS=$1; NC=$2
  echo "=== testing n-schedules=$NS n-collocation=$NC ==="
  /usr/bin/time -v python -m pinn_seir.fit_pinn \
    --census data/great_brittain/census.csv \
    --commute data/great_brittain/commute.csv \
    --crosswalk data/great_brittain/crosswalk.tsv \
    --contacts data/contacts \
    --flu data/epidemic/uk_flu_per_100000.csv \
    --adam-iters 30 \
    --dropout-rate 0.1 \
    --n-schedules "$NS" \
    --n-collocation "$NC" \
    --w-phys 100 \
    --out /scratch/mbarajas/epi-rl/logs/sizetest_${NS}_${NC} \
    2>&1 | tail -5 || echo "FAILED for $NS $NC"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
done
