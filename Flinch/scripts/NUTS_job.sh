#!/bin/bash

#SBATCH --account=rrg-wperciva
#SBATCH --job-name=FLINCH_NUTS
#SBATCH --output=NUTS_%A_%a.out
#SBATCH --time=7-00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --array=0-14
#SBATCH --mail-user=e34smith@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --constraint=genoa


module load julia/1.10.0

SEEDS = (4729 9352 7581 2053 6863)

SEED_INDEX = $((SLURM_ARRAY_TASK_ID / 3))
CHAIN_ID = $((SLURM_ARRAY_TASK_ID % 3 + 1))
SEED = ${SEEDS[SEED_INDEX]}

echo "======================="
echo "SLURM ARRAY JOB"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Chain ID: $CHAIN_ID"
echo "======================="

srun julia --project=/home/esmith/projects/rrg-wperciva/esmith/Flinch -t 32 /home/esmith/projects/rrg-wperciva/esmith/Flinch/LOO-PIT.jl $SEED $CHAIN_ID
