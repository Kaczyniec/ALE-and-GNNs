"""
Synthetic directed-graph generator + GNN link-prediction demo
Generative setup:
  - N nodes, each node has a feature vector x \in R^D.
  - One designated feature index `feat_idx` controls edge probability.
  - For any ordered pair (u->v) we draw an edge with
        p(edge u->v) = sigmoid(k * (x_u[feat_idx] - threshold))
    so edge probability depends only on the SOURCE node's feature.

This script builds a PyTorch Geometric dataset, a simple GNN link predictor
and provides a function to compute an ALE-like effect for the chosen feature
by perturbing the source node feature and measuring change in predicted
link probability for outgoing edges.

Notes:
  - The generative model intentionally isolates the causal effect of one
    feature on outgoing link probability (directed) so you have a ground truth
    relationship to compare ALE results against.
  - ALE for GNNs: when you change a node's feature you necessarily change the
    messages passed to neighbors during inference. This script performs the
    *interventional* approach: modify the node's feature and recompute
    forward passes. This is more faithful but more computationally expensive.

Requirements:
  - torch, torch_geometric (and its dependencies), scikit-learn, matplotlib

Usage: run the file to generate the dataset, train the GNN and compute ALE
curves for the selected feature.
"""
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
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

# ----------------------------- Utilities ----------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# -------------------------- Synthetic generator ---------------------------

def generate_directed_synthetic_graph(N=500, D=8, feat_idx=0,
                                      k=8.0, threshold=0.0,
                                      edge_sparsity=0.05, seed=42):
    """
    Generate node features and directed edges where the edge probability
    only depends on the source node's feature at `feat_idx`.

    Args:
      N: number of nodes
      D: node feature dimension
      feat_idx: which feature determines outgoing edge probability
      k, threshold: parameters of logistic p = sigmoid(k * (x - threshold))
      edge_sparsity: approx fraction of ordered pairs considered for edges
                    (to avoid N^2 explosion). For each source node we sample
                    about (edge_sparsity * N) candidate targets and
                    accept them with the Bernoulli probability.

    Returns: PyG Data object with directed edges stored in edge_index
             (shape [2, E]) and edge_label for training/evaluation
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Node features: sample standard normal, but make distribution wider for feat_idx
    X = np.random.normal(0, 1, size=(N, D)).astype(np.float32)
    # Optionally make the signal feature more spread to see effects
    X[:, feat_idx] = np.random.normal(0, 1.5, size=(N,))

    # Build edges
    sources = []
    targets = []
    labels = []

    num_candidates_per_node = max(5, int(edge_sparsity * N))
    for u in range(N):
        # choose candidate targets without replacement (excluding self)
        candidates = list(range(N))
        candidates.remove(u)
        sampled = np.random.choice(candidates, size=num_candidates_per_node, replace=False)
        p = sigmoid(k * (X[u, feat_idx] - threshold))
        for v in sampled:
            if np.random.rand() < p:
                # create directed edge u -> v
                sources.append(u)
                targets.append(v)
                labels.append(1)
            else:
                # negative sample (explicitly include with label 0)
                sources.append(u)
                targets.append(v)
                labels.append(0)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    x = torch.tensor(X, dtype=torch.float)
    edge_label = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_label=edge_label)
    data.N = N
    data.D = D
    data.feat_idx = feat_idx
    data.gen_k = k
    data.gen_threshold = threshold
    return data

# ---------------------------- GNN Model ----------------------------------

class LinkPredictorGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden=32, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index, edge_pairs):
        # x: [N, D]; edge_pairs: [2, E_pairs] listing (u,v) pairs we want probs for
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        # get node embeddings
        u_idx = edge_pairs[0]
        v_idx = edge_pairs[1]
        h_u = x[u_idx]
        h_v = x[v_idx]
        # concat for directed pair (u->v)
        h = torch.cat([h_u, h_v], dim=1)
        logits = self.mlp(h).squeeze(-1)
        return logits

# ---------------------------- Training -----------------------------------

def train_link_predictor(data, hidden=64, num_layers=2, epochs=30,
                         batch_size=2048, lr=1e-3, device='cpu'):
    device = torch.device(device)
    model = LinkPredictorGNN(data.D, hidden=hidden, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_label = data.edge_label.to(device)

    # We'll train in randomized mini-batches over the provided edges (which include both 1 and 0 labels)
    E = edge_index.size(1)
    indices = np.arange(E)

    for ep in range(epochs):
        model.train()
        np.random.shuffle(indices)
        losses = []
        for i in range(0, E, batch_size):
            batch_idx = indices[i:i+batch_size]
            pairs = edge_index[:, batch_idx].to(device)
            labels = edge_label[batch_idx].to(device)
            logits = model(x, edge_index, pairs)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}")

    return model

# ----------------------------- Evaluation --------------------------------

def evaluate_model_auc(model, data, device='cpu'):
    model.eval()
    device = torch.device(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_label = data.edge_label.cpu().numpy()
    with torch.no_grad():
        logits = model(x, edge_index, edge_index).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
    try:
        auc = roc_auc_score(edge_label, probs)
    except Exception:
        auc = float('nan')
    return auc

# ----------------------------- ALE for GNN -------------------------------

def compute_ale_for_feature(model, data, feat_idx, grid, device='cpu',
                            sample_node_indices=None):
    """
    Compute an interventional ALE-like curve for a node-level feature.
    We perturb the SOURCE node's feature and record how the predicted probability
    of its outgoing links changes (since our generative link probability depends
    only on the source feature, this targets that effect).

    Args:
      model: trained GNN
      data: PyG Data with x, edge_index, edge_label
      feat_idx: which feature to vary
      grid: 1D array of feature values to test
      sample_node_indices: optional subset of source nodes to average across.

    Returns:
      grid (np.array), ale_mean (np.array) - average predicted prob across perturbed nodes
    """
    model.eval()
    device = torch.device(device)
    x_orig = data.x.clone().to(device)
    edge_index = data.edge_index.to(device)
    N = x_orig.size(0)

    if sample_node_indices is None:
        sample_node_indices = list(range(N))

    probs_per_grid = []
    for val in grid:
        x = x_orig.clone()
        # set feature for sampled nodes to `val`
        x[sample_node_indices, feat_idx] = float(val)
        with torch.no_grad():
            logits = model(x, edge_index, edge_index).cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
        # For ALE we need to consider outgoing edges grouped by their source node.
        # We'll compute per-source mean predicted probability over the ordered pairs
        src_indices = edge_index[0].cpu().numpy()
        per_source = {}
        for idx_edge, src in enumerate(src_indices):
            per_source.setdefault(src, []).append(probs[idx_edge])
        per_source_mean = {s: np.mean(arr) for s, arr in per_source.items()}
        # average across sampled nodes
        vals = [per_source_mean[s] for s in sample_node_indices if s in per_source_mean]
        if len(vals) == 0:
            probs_per_grid.append(np.nan)
        else:
            probs_per_grid.append(np.mean(vals))

    return np.array(grid), np.array(probs_per_grid)

# --------------------------- Main demo -----------------------------------

def main():

    model = train_link_predictor(data, hidden=64, num_layers=2, epochs=40, device=device)
    auc = evaluate_model_auc(model, data, device=device)
    print(f"Test AUC (trained on synthetic edges): {auc:.4f}")

    # Compute ALE-like curve for feat_idx
    feat_idx = data.feat_idx
    # build a grid around empirical percentiles of that feature
    xs = data.x[:, feat_idx].cpu().numpy()
    grid = np.linspace(np.percentile(xs, 1), np.percentile(xs, 99), 21)
    grid, ale_vals = compute_ale_for_feature(model, data, feat_idx, grid, device=device,
                                            sample_node_indices=None)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(grid, ale_vals, marker='o')
    plt.title('Interventional ALE-like curve (predicted outgoing link prob)')
    plt.xlabel(f'Feature {feat_idx} value')
    plt.ylabel('Mean predicted outgoing link probability')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ale_curve.png')
    print('Saved ale_curve.png')

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)
    DATASET_NAME = "synthetic2"
    parser = argparse.ArgumentParser(
    )

    data = generate_directed_synthetic_graph(N=800, D=6, feat_idx=0, k=6.0, threshold=0.0, edge_sparsity=0.03, seed=123)
    print(f"Generated graph: N={data.N}, D={data.D}, edges={data.edge_index.size(1)}")

    device = 'cuda'
    parser.add_argument("--name", type=str, default=DATASET_NAME, help="Dataset name")
    parser.add_argument("--model_type", type=str, default="GCN", help="Architecture")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--column", type=int, default=0, help="Which column explain?")
    args = parser.parse_args()
    batch_size=64

    data_path = 'data/synthetic2/'

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
    for n in range(1):
        ale_approximate, t_approximate = accumulated_local_effects_approximate(model,train_data, column, 10, None, None, device)
        ale_exact, t_exact = accumulated_local_effects_exact(model,train_data, column, 10, None, None, device)

        df = pd.DataFrame({'trial': n, 'k': 2**k, 'max_bin_size': 2**max_bin_size, 'explanation_approximate': ale_approximate, 'time_approximate': t_approximate, 'explanation_exact': ale_exact, 'time_exact': t_exact}).reset_index()#.to_csv(os.path.join("data", args.name, f"ALE_synthetic_{args.model_type}_n_layers{args.n_layers}_hidden_size{args.hidden_dim}.csv"), header=False)
    

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    print(df)

    # Plot exact explanations
    for trial in df['trial'].unique():
        trial_data = df[df['trial'] == trial]
        axes[0].plot(trial_data['index'], trial_data['explanation_exact'], 
                    label=f'Trial {trial}', alpha=0.7)

    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Explanation Exact')
    axes[0].set_title('Exact Explanations - All Trials')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot approximate explanations
    for trial in df['trial'].unique():
        trial_data = df[df['trial'] == trial]
        axes[1].plot(trial_data['index'], trial_data['explanation_approximate'], 
                    label=f'Trial {trial}', alpha=0.7)

    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Explanation Approximate')
    axes[1].set_title('Approximate Explanations - All Trials')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    # results = pd.concat([results, pd.DataFrame({'idx': i, 'k': 2**k, 'max_bin_size': 2**max_bin_size, 'explanation_exact': ale_exact, 'time_exact': t_exact, 'explanation_approximate': ale_approximate, 'time_approximate': t_approximate})])
    #      
