#!/bin/bash
#SBATCH --job-name=scb_experiments
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=scb_experiments-%j.out
#SBATCH --error=scb_experiments-%j.err

# Load required modules
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate your uv-managed virtual environment
source .venv/bin/activate

# Run your experiments
python -m npsem.NIPS2018POMIS_exp.test_bandit_strategies