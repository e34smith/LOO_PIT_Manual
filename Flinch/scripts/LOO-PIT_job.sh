#!/bin/bash

#SBATCH --account=rrg-wperciva
#SBATCH --job-name=NUTS
#SBATCH --output=LOO-PIT.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-user=e34smith@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

module load julia/1.10.0

srun julia --project=/home/esmith/projects/rrg-wperciva/esmith/Flinch -t 32 /home/esmith/projects/rrg-wperciva/esmith/Flinch/scripts/LOO-PIT.jl
