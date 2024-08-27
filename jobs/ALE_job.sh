#!/bin/bash
#SBATCH --job-name=ale
#SBATCH --output=ale_mouse_output.log
#SBATCH --error=ale_mouse_error.log
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=20:00:00


# Activate virtual environment
source ./.venv/bin/activate

# Paths to the edges and node features CSV files
EDGES_PATH="data/CD1-E_no2/CD1-E-no2_iso3um_stitched_segmentation_bulge_size_3.0/CD1-E-no2_iso3um_stitched_segmentation_bulge_size_3.0_edges_processed.csv"
NODE_FEATURES_PATH="data/CD1-E_no2/CD1-E-no2_iso3um_stitched_segmentation_bulge_size_3.0/CD1-E-no2_iso3um_stitched_segmentation_bulge_size_3.0_nodes_processed.csv"
DATASET_NAME="CD1-E_no2"

# Define configurations
CONFIGS=(
    "--model_type GCN --hidden_dim 256 --n_layers 2"
    "--model_type GAT --hidden_dim 256 --n_layers 2"
)

# Loop through configurations and run the script
for CONFIG in "${CONFIGS[@]}"; do
    python3 ./utils/ALE.py \
        --edges_path $EDGES_PATH \
        --node_features_path $NODE_FEATURES_PATH \
        --name $DATASET_NAME \
        $CONFIG
done