#!/bin/bash
#SBATCH --job-name=Cnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END
# #SBATCH --mail-user=s.ciarella@esciencecenter.nl
#SBATCH --array=1-1

module load 2023
module load juliaup/1.14.5-GCCcore-12.3.0
# Note:
# - gpu_a100: 18 cores
# - gpu_h100: 16 cores
# https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting

mkdir -p /scratch-shared/$USER

echo "Slurm job ID: $SLURM_JOB_ID"
echo "Slurm array task ID: $SLURM_ARRAY_TASK_ID"

export CONF_FILE=$1

cd $HOME/CoupledNODE.jl/simulations/Benchmark

julia --project -t auto benchmark.jl

