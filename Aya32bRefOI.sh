#!/bin/bash
#SBATCH --job-name=Aya32bRefOI
#SBATCH --gres=gpu:rtx6000:4
#SBATCH --time=12:00:00
#SBATCH -c 16                  # Adjust number of cores as needed
#SBATCH --mem=100G             # Adjust memory if needed
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Stop on error
set -e

# Load necessary modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Go to script directory (optional but recommended)
cd /scratch/ssd004/scratch/haigelee/Mitacs/AYA


# Run the commands
# echo "[$(date)] Running single presence with brief prompt mode"
# python Aya32b.py --split single_presence --prompt_mode brief

# echo "[$(date)] Running co-occurrence with brief prompt mode"
# python Aya32b.py --split co_occurrence --prompt_mode brief

# echo "[$(date)] Running single presence with default prompt mode"
# python Aya32b.py --split single_presence --prompt_mode default

echo "[$(date)] Running co-occurrence with default prompt mode 32b"
python Aya32b.py --split co_occurrence --prompt_mode default

echo "[$(date)] Running co-occurrence with default prompt mode 8b"
python Aya8b.py --split co_occurrence --prompt_mode default

echo "[$(date)] All done!"
#sbatch Aya32bRefOI.sh