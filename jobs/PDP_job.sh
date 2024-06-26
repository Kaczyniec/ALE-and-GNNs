#!/bin/bash
#SBATCH --job-name=pdp
#SBATCH --output=pdp_output_17_06.log
#SBATCH --error=pdp_error_17_06.log
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=20:00:00


# Activate virtual environment
source ./.venv/bin/activate

# Run the script
python ./utils/PDP.py