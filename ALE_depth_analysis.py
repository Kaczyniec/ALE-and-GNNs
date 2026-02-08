import sys
import os
import logging
import gc
import torch
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

# Append current working directory to path for local module resolution
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from utils.preprocess_data import graph_data
from models.gnn_batchnorm import Model, train, test
from utils.ALE import (
    accumulated_local_effects_exact,
    accumulated_local_effects_approximate,
)

# Configuration
DATA_DIR = os.path.join("data", "citations")
MODEL_DIR = os.path.join("models", "citations")
LOG_DIR = os.path.join(MODEL_DIR, "depth")

os.makedirs(MODEL_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on: ", device)
    
    edges_path = os.path.join(DATA_DIR, "edge.parquet")
    node_features_path = os.path.join(DATA_DIR, "node_features.parquet")
    
    logger = logging.getLogger(__name__)

    edges = pd.read_parquet(edges_path)
    node_features = pd.read_parquet(node_features_path)
    
    train_loader, test_loader, train_data, test_data, val_data, val_loader = graph_data(
        edges, node_features, os.path.join(DATA_DIR, "")
    )

    for no_exp in range(2, 10):
        for n_layers in [2, 3, 4, 5]:
            hidden_channels = int(512 / n_layers)
            
            model_name = f"GAT,n_layers{n_layers},hidden_size{hidden_channels},no{no_exp}"
            model_path = os.path.join(MODEL_DIR, model_name)

            model = Model(
                in_channels=np.shape(train_data.x)[1],
                hidden_channels=hidden_channels,
                model_type="GAT",
                n_layers=n_layers,
            ).to(device)

            print(model_path)
            
            if os.path.isfile(model_path):
                model.load_state_dict(torch.load(model_path))
                print("Model weights loaded successfully.")
            else:
                # task = Task.init(
                #    project_name="Citation_Training",
                #    task_name=f"GAT,n_layers{n_layers},hidden_size{hidden_channels}",
                # )
                print("Model weights file does not exist. Initializing model with random weights.")
                
                optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-6)
                loss_values = []
                
                for epoch in range(1, 5 * n_layers):
                    logging.info(f"Starting epoch {epoch}")
                    loss = train(train_loader, device, optimizer, model, writer, epoch)
                    f1 = test(model, test_loader)
                    loss_values.append(loss)
                
                torch.save(model.state_dict(), model_path)

            max_bin_size = 10
            n = 10

            for trial in range(3):
                ale_approximate, t_approximate = accumulated_local_effects_approximate(
                    model, train_data, 0, 5, 2**max_bin_size, k=2**n
                )
                ale_exact, t_exact = accumulated_local_effects_exact(
                    model, train_data, 0, 5, 2**max_bin_size, k=2**n
                )

                result_data = {
                    "k": 2**n,
                    "max_bin_size": 2**max_bin_size,
                    "ALE exact": ale_exact,
                    "time_exact": t_exact,
                    "ALE approximate": ale_approximate,
                    "time_approximate": t_approximate,
                    "n_layers": n_layers,
                    "hidden_size": hidden_channels,
                    "no_exp": no_exp,
                    "trial": trial,
                }
                
                csv_path = os.path.join(DATA_DIR, "ALE_depth_GAT3.csv")
                pd.DataFrame(result_data).to_csv(
                    csv_path,
                    mode="a",
                    header=False,
                )

            # task.close()
            del model
            gc.collect()
            torch.cuda.empty_cache()