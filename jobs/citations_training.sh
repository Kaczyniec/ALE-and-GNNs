#!/bin/bash
#SBATCH --job-name=citation_gnn
#SBATCH --output=citation_gnn_output.log
#SBATCH --error=citation_gnn_error.log
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00


# Activate virtual environment
source ./.venv/bin/activate

# Run the script
python ./models/twitch_gnn.py