#!/bin/bash
#SBATCH --job-name=pinn_size_64_128
#SBATCH --account=def-divya03
#SBATCH --partition=debug
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=00:15:00
#SBATCH --chdir=/scratch/mbarajas/epi-rl
#SBATCH --output=/scratch/mbarajas/epi-rl/logs/%x-%j.out
#SBATCH --error=/scratch/mbarajas/epi-rl/logs/%x-%j.out

set -uo pipefail
cd /scratch/mbarajas/epi-rl
module load python/3.11.5
source .venv/bin/activate
export DDE_BACKEND=pytorch

python -m pinn_seir.fit_pinn \
  --census data/great_brittain/census.csv \
  --commute data/great_brittain/commute.csv \
  --crosswalk data/great_brittain/crosswalk.tsv \
  --contacts data/contacts \
  --flu data/epidemic/uk_flu_per_100000.csv \
  --adam-iters 20 \
  --dropout-rate 0.1 \
  --n-schedules 64 \
  --n-collocation 128 \
  --w-phys 100 \
  --out /scratch/mbarajas/epi-rl/logs/sizetest_64_128_v2
echo "EXIT:$?"
