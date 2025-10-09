import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm
import sys
import argparse
import os

from torch_geometric.utils import k_hop_subgraph

from utils.preprocess_data import graph_data
from models.gnn_batchnorm import Model 


def accumulated_local_effects_exact(
    model,
    dataset,
    feature_index,
    num_bins=10,
    max_bin_size=None,
    k=256,
    use_khop=False,
    khop_size=4,
    device=torch.device("cuda"),
):
    start = time.time()
    feature_values = dataset.x[:, feature_index].cpu()
    bin_edges = np.linspace(
        feature_values.min() - 0.0001, feature_values.max() + 0.0001, num_bins + 1
    )
    bin_indices = np.digitize(feature_values.numpy(), bin_edges) - 1
    ale = []
    nodes = set(range(dataset.x.shape[0]))
    
    with torch.no_grad():
        for bin_idx in range(num_bins):
            bin_ale = []
            bin_data_idx = np.where(bin_indices == bin_idx)[0]
            if bin_data_idx.shape[0] == 0 and len(ale) > 0:
                ale.append(ale[-1])
                continue
            if max_bin_size != None:
                bin_data_idx_subset = np.random.choice(
                    bin_data_idx,
                    size=min(max_bin_size, bin_data_idx.shape[0]),
                    replace=False,
                )
            else:
                bin_data_idx_subset = bin_data_idx
            
            for idx in tqdm(bin_data_idx_subset):
                data = dataset.clone()
                data.to(device)  
                model.eval()
                
                if k is not None:
                    unique = np.random.choice(list(nodes), size=k)
                else:
                    unique = np.array(list(nodes))
                
                if use_khop:
                    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                        [idx] + list(unique), khop_size, data.edge_index, relabel_nodes=True
                    )
                    
                    if len(mapping) < k + 1:
                        edge_label_index = torch.cat(
                            (
                                mapping[0].repeat(len(mapping) - 1).unsqueeze(-1),
                                mapping[1:].long().unsqueeze(-1),
                            ),
                            dim=1,
                        ).T.long().to(device)
                    else:
                        edge_label_index = torch.cat(
                            (
                                mapping[0].repeat(k).unsqueeze(-1),
                                mapping[1:k+1].long().unsqueeze(-1),
                            ),
                            dim=1,
                        ).T.long().to(device)
                    
                    data.edge_index = edge_index.to(device)
                    
                    data.x[idx, feature_index] = torch.tensor(bin_edges[bin_idx], device=device, dtype=torch.float32)
                    lower_encode = model(data.x[subset], data.edge_index)
                    lower = model.decode(lower_encode, edge_label_index).view(-1).sigmoid()
                    
                    data.x[idx, feature_index] = torch.tensor(bin_edges[bin_idx + 1], device=device, dtype=torch.float32)
                    upper_encode = model(data.x[subset], data.edge_index)
                    upper = model.decode(upper_encode, edge_label_index).view(-1).sigmoid()
                else:
                    edge_label_index = torch.cat(
                        (
                            torch.full((k,), idx, dtype=torch.long, device=device).unsqueeze(-1),
                            torch.tensor(unique, dtype=torch.long, device=device).unsqueeze(-1),
                        ),
                        dim=1,
                    ).T.long()
                    print(edge_label_index)
                    
                    data.x[idx, feature_index] = torch.tensor(bin_edges[bin_idx], device=device, dtype=torch.float32)
                    lower_encode = model(data.x, data.edge_index)
                    lower = model.decode(lower_encode, edge_label_index).view(-1).sigmoid()
                    
                    data.x[idx, feature_index] = torch.tensor(bin_edges[bin_idx + 1], device=device, dtype=torch.float32)
                    upper_encode = model(data.x, data.edge_index)
                    upper = model.decode(upper_encode, edge_label_index).view(-1).sigmoid()
                
                bin_ale.append(float(torch.mean((upper - lower)).cpu().detach()))
            
            bin_ale = np.mean(bin_ale)
            if ale:
                ale.append(bin_ale + ale[-1])
            else:
                ale.append(bin_ale)
    
    end = time.time()
    return ale, end - start

def accumulated_local_effects_approximate(
    model,
    dataset,
    feature_index,
    num_bins=10,
    max_bin_size=None,
    k=256,
    use_khop=False,
    khop_size=4,
    device=torch.device("cuda"),
):
    start = time.time()
    # Step 1: Divide the range of values of the selected variable into bins
    feature_values = dataset.x[:, feature_index].cpu()
    bin_edges = np.linspace(
        feature_values.min() - 0.0001, feature_values.max() + 0.0001, num_bins + 1
    )
    # Step 2: Sort the data into these bins according to the value of the feature
    bin_indices = np.digitize(feature_values.numpy(), bin_edges) - 1
    print(bin_edges)
    # Initialize arrays to store predictions and accumulated local effects
    ale = []
    nodes = set(range(dataset.x.shape[0]))
    # Step 3-5: Calculate accumulated local effects for each bin
    with torch.no_grad():
        for bin_idx in range(num_bins):
            # Filter dataset based on bin index
            data = dataset.clone()
            bin_data_idx = np.where(bin_indices == bin_idx)[0]
            if bin_data_idx.shape[0] == 0 and len(ale) > 0:
                ale.append(ale[-1])
                continue
            if max_bin_size != None:
                bin_data_idx_subset = np.random.choice(
                    bin_data_idx,
                    size=min(max_bin_size, bin_data_idx.shape[0]),
                    replace=False,
                )
            else:
                bin_data_idx_subset = bin_data_idx
            
            model.eval()
            # Step 3: Calculate model predictions for lower and upper end of the section
            if k is not None:
                unique = np.random.choice(list(nodes), size=k)
            else:
                unique = np.array(list(nodes))
            
            if use_khop:
                # Use k-hop subgraph extraction
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                    list(bin_data_idx_subset) + list(unique),
                    khop_size,
                    data.edge_index,
                    relabel_nodes=True,
                )
                
                data.edge_index = edge_index
                data.to(device)
                edge_index = edge_index.to(device)
                unique_tensor = torch.Tensor(unique).int()
                
                # Create edge label index: all bin nodes to all unique nodes
                edge_label_index = torch.cat(
                    (
                        torch.Tensor(mapping[:len(bin_data_idx_subset)])
                        .int()
                        .repeat_interleave(unique_tensor.shape[0])
                        .unsqueeze(-1),
                        mapping[len(bin_data_idx_subset):]
                        .repeat(bin_data_idx_subset.shape[0])
                        .int()
                        .unsqueeze(-1),
                    ),
                    dim=1,
                ).T.long()
                
                # Lower bound prediction
                data.x[bin_data_idx_subset, feature_index] = torch.tensor(
                    bin_edges[bin_idx], device=device
                ).float()
                lower_encode = model(data.x[subset], edge_index)
                lower = model.decode(lower_encode, edge_label_index).view(-1).sigmoid()
                
                # Upper bound prediction
                data.x[bin_data_idx_subset, feature_index] = torch.tensor(
                    bin_edges[bin_idx + 1], device=device
                ).float()
                upper_encode = model(data.x[subset], edge_index)
                upper = model.decode(upper_encode, edge_label_index).view(-1).sigmoid()
            else:
                # Use full graph
                data.to(device)
                
                # Create edge label index: all bin nodes to all unique nodes (Cartesian product)
                bin_nodes_tensor = torch.tensor(bin_data_idx_subset, dtype=torch.long)
                unique_tensor = torch.tensor(unique, dtype=torch.long)
                
                edge_label_index = torch.cat(
                    (
                        bin_nodes_tensor.repeat_interleave(unique_tensor.shape[0]).unsqueeze(-1),
                        unique_tensor.repeat(bin_data_idx_subset.shape[0]).unsqueeze(-1),
                    ),
                    dim=1,
                ).T.long()
                
                edge_label_index = edge_label_index.to(device)
                
                # Lower bound prediction
                data.x[bin_data_idx_subset, feature_index] = torch.tensor(
                    bin_edges[bin_idx], device=device
                ).float()
                lower_encode = model(data.x, data.edge_index)
                lower = model.decode(lower_encode, edge_label_index).view(-1).sigmoid()
                
                # Upper bound prediction
                data.x[bin_data_idx_subset, feature_index] = torch.tensor(
                    bin_edges[bin_idx + 1], device=device
                ).float()
                upper_encode = model(data.x, data.edge_index)
                upper = model.decode(upper_encode, edge_label_index).view(-1).sigmoid()
            
            # Step 4: Subtract the above values. Average across all data points in the bin
            if ale:
                ale.append(float(torch.mean(upper - lower).detach()) + ale[-1])
            else:
                ale.append(float(torch.mean(upper - lower).detach()))
            del data
    
    end = time.time()
    return ale, end - start

# Change: the prediction is between node with change value and the rest of the dataset!!!
def accumulated_local_effects_exact_no_subgraph(
    model,
    dataset,
    feature_index,
    num_bins=10,
    max_bin_size=None,
    k=256,
    device=torch.device("cuda"),
):
    start = time.time()
    # Step 1: Divide the range of values of the selected variable into bins
    feature_values = dataset.x[:, feature_index].cpu()
    bin_edges = np.linspace(
        feature_values.min() - 0.001, feature_values.max() + 0.001, num_bins + 1
    )
    # Step 2: Sort the data into these bins according to the value of the feature
    bin_indices = np.digitize(feature_values.numpy(), bin_edges) - 1
    # Initialize arrays to store predictions and accumulated local effects
    ale = []
    nodes = set(range(dataset.x.shape[0]))
    # Step 3-5: Calculate accumulated local effects for each bin
    with torch.no_grad():
        for bin_idx in range(num_bins):
            # Filter dataset based on bin index
            bin_ale = []
            bin_data_idx = np.where(bin_indices == bin_idx)[0]
            if bin_data_idx.shape[0] == 0 and len(ale) > 0:
                ale.append(ale[-1])
                continue
            if max_bin_size != None:
                bin_data_idx_subset = np.random.choice(
                    bin_data_idx,
                    size=min(max_bin_size, bin_data_idx.shape[0]),
                    replace=False,
                )
            else:
                bin_data_idx_subset = bin_data_idx
            for idx in tqdm(bin_data_idx_subset):
                data = dataset.clone()
                model.eval()
                # Step 3: Calculate model predictions for lower and upper end of the section
                unique = np.random.choice(list(nodes), size=k)
                
                # Create edge label index for predictions: idx to all unique nodes
                edge_label_index = torch.cat(
                    (
                        torch.full((k,), idx, dtype=torch.long).unsqueeze(-1),
                        torch.tensor(unique, dtype=torch.long).unsqueeze(-1),
                    ),
                    dim=1,
                ).T.long()
                
                data.to(device)
                edge_label_index = edge_label_index.to(device)
                
                # Lower bound prediction
                data.x[idx, feature_index] = torch.tensor(bin_edges[bin_idx], device=device).float()
                #data.x[idx, feature_index] = torch.tensor(bin_edges[bin_idx], device=device).float()
                lower_encode = model(data.x, data.edge_index)
                lower = model.decode(lower_encode, edge_label_index).view(-1).sigmoid()
                
                # Upper bound prediction
                data.x[idx, feature_index] = torch.tensor(bin_edges[bin_idx + 1], device=device).float()
                upper_encode = model(data.x, data.edge_index)
                upper = model.decode(upper_encode, edge_label_index).view(-1).sigmoid()
                
                # Step 4: Subtract the above values. Average across all data points in the bin
                bin_ale.append(float(torch.mean((upper - lower)).detach()))
            
            # Step 5: Calculate the cumulative sum of the averaged differences
            bin_ale = np.mean(bin_ale)
            if ale:
                ale.append(bin_ale + ale[-1])
            else:
                ale.append(bin_ale)
    
    end = time.time()
    return ale, end - start



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)
    EDGES_PATH = "data/citations/edge.csv"#"data/CD1-E_no2/CD1-E-no2_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"  # 
    NODE_FEATURES_PATH = "data/citations/node_features.csv"#"data/CD1-E_no2/CD1-E-no2_iso3um_stitched_segmentation_bulge_size_3.0_nodes.csv"  # 
    DATASET_NAME = "CD1-E_no2"#citations"#
    parser = argparse.ArgumentParser(
    )
    parser.add_argument(
        "--edges_path", type=str, default=EDGES_PATH, help="Path to the edges CSV file"
    )
    parser.add_argument(
        "--node_features_path",
        type=str,
        default=NODE_FEATURES_PATH,
        help="Path to the node features CSV file",
    )
    parser.add_argument("--name", type=str, default=DATASET_NAME, help="Dataset name")
    parser.add_argument("--model_type", type=str, default="GCN", help="Architecture")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--column", type=int, default=0, help="Which column explain?")
    args = parser.parse_args()

    #args.edges_path = 'data/citations/edge.parquet'
    #args.node_features_path = 'data/citations/node_features.parquet'
    
    #args.name='citations'
    edges = pd.read_csv(EDGES_PATH, index_col=0, sep=";")[["node1id", "node2id"]]
    node_features = pd.read_csv(NODE_FEATURES_PATH, sep=";")[["pos_x", "pos_y", "pos_z", "isAtSampleBorder"]]
    #edges = pd.read_parquet(args.edges_path)
    #node_features = pd.read_parquet(args.node_features_path)

    data_path = 'data/citations/'
    train_loader, test_loader, train_data, test_data = graph_data(
        edges, node_features, "data/" + args.name
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
    # 
    for k in range(4, 11):
      for max_bin_size in range(4, 11):
        #    print(k, max_bin_size)

        ale_approximate, t_approximate = accumulated_local_effects_approximate(model,train_data, args.column, 5, 2**max_bin_size, 2**k, device)
        ale_exact, t_exact = accumulated_local_effects_exact(model,train_data, args.column, 5, 2**max_bin_size, 2**k, device)

        pd.DataFrame({'k': 2**k, 'max_bin_size': 2**max_bin_size, 'explanation_approximate': ale_approximate, 'time_approximate': t_approximate, 'explanation_exact': ale_exact, 'time_exact': t_exact}).to_csv(os.path.join("data", args.name, f"ALE_04.08_{args.model_type}_n_layers{args.n_layers}_hidden_size{args.hidden_dim}.csv"), mode='a', header=False)
        
    # results = pd.concat([results, pd.DataFrame({'idx': i, 'k': 2**k, 'max_bin_size': 2**max_bin_size, 'explanation_exact': ale_exact, 'time_exact': t_exact, 'explanation_approximate': ale_approximate, 'time_approximate': t_approximate})])
    #        
