import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm
import sys
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from torch_geometric.utils import k_hop_subgraph

import torch_geometric.transforms as T
from utils.preprocess_data import graph_data
from models.gnn_batchnorm import Model 
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce

from utils.ALE import accumulated_local_effects_approximate, accumulated_local_effects_exact
import os 

from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from torch_geometric.loader import LinkNeighborLoader

from sklearn.preprocessing import normalize, MultiLabelBinarizer

def synthetic_edges(data, threshold = 0.5, p_high = 0.9, p_low = 0.1, k = 5):

    random_var = data.x[:, -1]
    num_nodes = data.num_nodes


    src, dst = data.edge_index
    z_src = random_var[src]

    probs = torch.where(z_src > threshold, 
                        torch.full_like(z_src, p_high), 
                        torch.full_like(z_src, p_low))

    mask = torch.rand_like(probs) < probs
    edge_index = data.edge_index[:, mask]

    # --- Generate candidate new edges ---
    num_candidates = k * num_nodes
    cand_src = torch.randint(0, num_nodes, (num_candidates,))
    cand_dst = torch.randint(0, num_nodes, (num_candidates,))

    z_src_cand = random_var[cand_src]
    probs_cand = torch.where(z_src_cand > threshold, 
                            torch.full_like(z_src_cand, p_high), 
                            torch.full_like(z_src_cand, p_low))

    mask_cand = torch.rand_like(probs_cand) < probs_cand
    cand_edges = torch.stack([cand_src[mask_cand], cand_dst[mask_cand]])

    # --- Merge original+new edges and clean up ---
    edge_index = torch.cat([edge_index, cand_edges], dim=1)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

    data.edge_index = edge_index
    return data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)
    EDGES_PATH = "data/citations/edge.parquet"#"data/CD1-E_no2/CD1-E-no2_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"  # 
    NODE_FEATURES_PATH = "data/citations/node_features.parquet"#"data/CD1-E_no2/CD1-E-no2_iso3um_stitched_segmentation_bulge_size_3.0_nodes.csv"  # 
    DATASET_NAME = "synthetic1"
    parser = argparse.ArgumentParser(
    )

    parser.add_argument("--name", type=str, default=DATASET_NAME, help="Dataset name")
    parser.add_argument("--model_type", type=str, default="GAT", help="Architecture")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--column", type=int, default=0, help="Which column explain?")
    args = parser.parse_args()
    batch_size=64
    #args.edges_path = 'data/citations/edge.parquet'
    #args.node_features_path = 'data/citations/node_features.parquet'
    edges = pd.read_parquet(EDGES_PATH)
    node_features = pd.read_parquet(NODE_FEATURES_PATH)
    node_features['random_var'] = np.random.rand(len(node_features))

    data_path = 'data/synthetic1/'
    edge_array = edges.values.T
    edge_index = torch.tensor(edge_array, dtype=torch.int64)
    data = Data(x=torch.tensor(node_features.values.astype(np.float32)),
            edge_index=edge_index,
            #y=node_features.company,
            )
    data = synthetic_edges(data)
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        #disjoint_train_ratio=0.3,
        #neg_sampling_ratio=2.0,
        add_negative_train_samples=False
    )

    train_data, val_data, test_data = transform(data)

    # Define seed edges:
    edge_label_index = train_data.edge_label_index
    edge_label = train_data.edge_label

    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=[30] * 2,
        batch_size=batch_size,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        #time_attr=, 
    )
    
    test_loader = LinkNeighborLoader(
        test_data,
        num_neighbors=[30] * 2,
        batch_size=batch_size,
        edge_label_index=test_data.edge_label_index,
        edge_label=test_data.edge_label,
        #neg_sampling_ratio=1.0,
    )
    val_loader = LinkNeighborLoader(
        test_data,
        num_neighbors=[30] * 2,
        batch_size=batch_size,
        edge_label_index=test_data.edge_label_index,
        edge_label=test_data.edge_label,
        #neg_sampling_ratio=1.0,
    )

    # model_path = f"models/citations/{config['model_type']},n_layers{config['n_layers']},hidden_size{config['hidden_channels']}"
    model_path = os.path.join(
        "models",
        args.name,
        args.model_type
        + ",n_layers"
        + str(args.n_layers)
        + ",hidden_size"
        + str(args.hidden_dim),
    )
    # Initialize the model
    model = Model(
        in_channels=np.shape(train_data.x)[1],
        hidden_channels=args.hidden_dim,
        model_type=args.model_type,
        n_layers=args.n_layers,
    ).to(device)

    # Check if the model weights file exists
    if os.path.isfile(model_path):
        # Load the weights into the model
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully from path:", model_path)
    else:
        print(
            "Model weights file does not exist. Initializing model with random weights."
        )

    # print(pd.DataFrame({'k': 512 , 'max_bin_size': None, 'explanation_exact': ale_approximate, 'time_exact': t_approximate}))#.to_csv(os.path.join("data", args.name, f"ALE_goldstandard_{args.model_type}_n_layers{args.n_layers}_hidden_size{args.hidden_dim}_no_k.csv"), mode='a', header=False)
    # results = pd.DataFrame(columns=['idx', 'k', 'max_bin_size', 'explanation_exact', 'time_exact', 'explanation_approximate', 'time_approximate'])
    # 4-11, 4-7; 7-11, 7-11
    column = -1
    
    k = 10
    max_bin_size = 10
    for n in range(10):
        ale_approximate, t_approximate = accumulated_local_effects_approximate(model,train_data, column, 5, 2**max_bin_size, 2**k, device)
        ale_exact, t_exact = accumulated_local_effects_exact(model,train_data, column, 5, 2**max_bin_size, 2**k, device)

        pd.DataFrame({'k': 2**k, 'max_bin_size': 2**max_bin_size, 'explanation_approximate': ale_approximate, 'time_approximate': t_approximate, 'explanation_exact': ale_exact, 'time_exact': t_exact}).to_csv(os.path.join("data", args.name, f"ALE_synthetic_{args.model_type}_n_layers{args.n_layers}_hidden_size{args.hidden_dim}.csv"), mode='a', header=False)
        
    # results = pd.concat([results, pd.DataFrame({'idx': i, 'k': 2**k, 'max_bin_size': 2**max_bin_size, 'explanation_exact': ale_exact, 'time_exact': t_exact, 'explanation_approximate': ale_approximate, 'time_approximate': t_approximate})])
    #      