#!/bin/bash
#SBATCH --job-name=julia_test
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --output=test_on_snellius.out

module load 2023
module load juliaup/1.14.5-GCCcore-12.3.0

srun --unbuffered julia --project -t auto -e 'using Pkg; Pkg.resolve(); Pkg.test()'

