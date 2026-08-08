#!/bin/bash
#SBATCH --job-name=pinn_final_sizing
#SBATCH --account=def-divya03
#SBATCH --partition=debug
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=00:30:00
#SBATCH --chdir=/scratch/mbarajas/epi-rl
#SBATCH --output=/scratch/mbarajas/epi-rl/logs/%x-%j.out
#SBATCH --error=/scratch/mbarajas/epi-rl/logs/%x-%j.out

set -uo pipefail
cd /scratch/mbarajas/epi-rl
module load python/3.11.5
source .venv/bin/activate
export DDE_BACKEND=pytorch

( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader; sleep 3; done ) > /scratch/mbarajas/epi-rl/logs/memlog_final.txt &
MEMPID=$!

echo "=== 48/96, 30 iters (fit check) ==="
python -m pinn_seir.fit_pinn \
  --census data/great_brittain/census.csv --commute data/great_brittain/commute.csv \
  --crosswalk data/great_brittain/crosswalk.tsv --contacts data/contacts \
  --flu data/epidemic/uk_flu_per_100000.csv \
  --adam-iters 30 --dropout-rate 0.1 --n-schedules 48 --n-collocation 96 --w-phys 100 \
  --out /scratch/mbarajas/epi-rl/logs/sizetest_48_96
echo "EXIT_48_96:$?"

echo "=== 32/64, 150 iters (throughput + stability check) ==="
python -m pinn_seir.fit_pinn \
  --census data/great_brittain/census.csv --commute data/great_brittain/commute.csv \
  --crosswalk data/great_brittain/crosswalk.tsv --contacts data/contacts \
  --flu data/epidemic/uk_flu_per_100000.csv \
  --adam-iters 150 --dropout-rate 0.1 --n-schedules 32 --n-collocation 64 --w-phys 100 \
  --out /scratch/mbarajas/epi-rl/logs/sizetest_32_64_long
echo "EXIT_32_64:$?"

kill $MEMPID 2>/dev/null
echo "=== peak mem ==="
sort -n /scratch/mbarajas/epi-rl/logs/memlog_final.txt | tail -3
