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
from torch_geometric.nn import SAGEConv, global_mean_pool, GCNConv, GATConv
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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from torch_geometric.utils import k_hop_subgraph
from torch.optim.lr_scheduler import ExponentialLR
import torch_geometric.transforms as T
from utils.preprocess_data import graph_data
from models.gnn_batchnorm import Model, train, test
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce

from utils.ALE import (
    accumulated_local_effects_approximate,
    accumulated_local_effects_exact,
    ALE_approximate_minimal,
    ALE_exact_minimal,
)
import os

from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from torch_geometric.loader import LinkNeighborLoader
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter("models/citations/tensorboard_logs2")

# ----------------------------- Utilities ----------------------------------


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# -------------------------- Synthetic generator ---------------------------


def generate_directed_synthetic_graph(
    N=500, D=8, feat_idx=0, k=8.0, threshold=0.0, edge_sparsity=0.05, seed=42
):
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
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # Node features: sample standard normal, but make distribution wider for feat_idx
    X = np.random.normal(0, 1, size=(N, D)).astype(np.float32)
    # Optionally make the signal feature more spread to see effects
    X[:, feat_idx] = np.random.uniform(-1, 1, size=(N,))

    # Build edges
    sources = []
    targets = []
    labels = []

    num_candidates_per_node = int(edge_sparsity * N)
    for u in range(N):
        # choose candidate targets without replacement (excluding self)
        candidates = list(range(N))
        candidates.remove(u)
        sampled = np.random.choice(
            candidates, size=num_candidates_per_node, replace=False
        )
        prob = (X[u, feat_idx]+X[sampled, feat_idx]+2)/4#np.exp(X[u, feat_idx]-X[sampled, feat_idx])/np.exp(2)-np.exp(-1)#sigmoid(k * (X[u, feat_idx]))# - X[sampled, feat_idx]
        for v, p in zip(sampled,prob):
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
    print(edge_index[:, labels].shape)
    data = Data(x=x, edge_index=edge_index[:, labels], edge_label_index=edge_index, edge_label=edge_label)
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
        self.convs.append(GATConv(in_channels, hidden))#SAGEConv
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden, hidden))
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, 1)#hidden), nn.ReLU(), nn.Linear(hidden, 1)
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

class LinkPredictorGNN2(torch.nn.Module):
    def __init__(self, in_channels, hidden=32, num_layers=2, dropout=0.3, heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        
        # First layer
        self.convs.append(GATConv(in_channels, hidden, heads=heads, concat=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden * heads))
        
        # Last layer (average heads instead of concat)
        if num_layers > 1:
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads, concat=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden))
        
        # Edge predictor MLP with more capacity
        final_hidden = hidden
        self.mlp = nn.Sequential(
            nn.Linear(final_hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, x, edge_index, edge_pairs):
        # GNN layers with residual connections (optional)
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x_new
        
        # Get node embeddings for pairs
        u_idx = edge_pairs[0]
        v_idx = edge_pairs[1]
        h_u = x[u_idx]
        h_v = x[v_idx]
        
        # Concat for directed pair (u->v)
        h = torch.cat([h_u, h_v], dim=1)
        logits = self.mlp(h).squeeze(-1)
        return logits
# ---------------------------- Training -----------------------------------


def train_link_predictor(
    data, hidden=64, num_layers=2, epochs=30, batch_size=2048, lr=5e-3, device="cpu"
):
    device = torch.device(device)
    model = LinkPredictorGNN(data.D, hidden=hidden, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_label = data.edge_label.to(device)
    edge_label_index = data.edge_label_index.to(device)

    # We'll train in randomized mini-batches over the provided edges (which include both 1 and 0 labels)
    E = edge_index.size(1)
    indices = np.arange(E)

    for ep in range(epochs):
        model.train()
        np.random.shuffle(indices)
        losses = []
        for i in range(0, E, batch_size):
            batch_idx = indices[i : i + batch_size]
            pairs = edge_label_index[:, batch_idx].to(device)#edge_index[:, batch_idx].to(device)
            labels = edge_label[batch_idx].to(device)

            logits = model(x, edge_index, pairs)
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}")

    return model


# ----------------------------- Evaluation --------------------------------


def evaluate_model_auc(model, data, device="cpu"):
    model.eval()
    device = torch.device(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_index_label = data.edge_label_index.to(device)
    edge_label = data.edge_label.cpu().numpy()
    with torch.no_grad():
        logits = model(x, edge_index, edge_index_label).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
    try:
        auc = roc_auc_score(edge_label, probs)
        print(edge_index.shape, edge_label.shape)
    except Exception:
        auc = float("nan")
    return auc


# split
from torch_geometric.utils import negative_sampling

# ----------------------------- ALE for GNN -------------------------------
def PDP_exact(
    model,
    dataset,
    feature_index,
    num_bins=10,
    max_bin_size=None,
    k=256,
    device=torch.device("cuda"),
):
    start = time.time()
    feature_values = dataset.x[:, feature_index].cpu()
    bin_edges = np.linspace(
        feature_values.min() - 0.001, feature_values.max() + 0.001, num_bins + 1
    )
    
    pdp = []
    
    with torch.no_grad():
        for bin_idx in range(num_bins):
            # For PDP, we don't care about which nodes are in this bin
            # We set ALL nodes' feature to the bin center value
            
            # Use bin center instead of edges
            bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
            
            # Get all node indices (or subsample if max_bin_size specified)
            all_node_idx = np.arange(dataset.x.shape[0])
            if max_bin_size is not None:
                all_node_idx = np.random.choice(
                    all_node_idx,
                    size=min(max_bin_size, len(all_node_idx)),
                    replace=False,
                )
            
            bin_predictions = []
            
            # Sample k random target nodes
            sample_nodes = np.random.choice(dataset.x.shape[0], size=k, replace=False)
            
            for idx in all_node_idx:
                data = dataset.clone()
                data.to(device)
                
                # Edge label index: from idx to all sampled nodes
                edge_label_index = torch.stack([
                    torch.full((k,), idx, dtype=torch.long),
                    torch.tensor(sample_nodes, dtype=torch.long)
                ]).to(device)
                
                # Set THIS node's feature to bin center (not all nodes!)
                data.x[idx, feature_index] = torch.tensor(
                    bin_center, device=device, dtype=torch.float32
                )
                
                # Get prediction
                logits = model(data.x, data.edge_index, edge_label_index).cpu().numpy()
                predictions = 1 / (1 + np.exp(-logits))
                
                # Store average prediction for this source node
                bin_predictions.append(np.mean(predictions))
            
            # PDP value is the average prediction across all source nodes
            pdp.append(np.mean(bin_predictions))
    
    end = time.time()
    return pdp, end - start


def PDP_approximate(
    model,
    dataset,
    feature_index,
    num_bins=10,
    max_bin_size=None,
    k=256,
    device=torch.device("cuda"),
):
    start = time.time()
    feature_values = dataset.x[:, feature_index].cpu()
    bin_edges = np.linspace(
        feature_values.min() - 0.001, feature_values.max() + 0.001, num_bins + 1
    )
    
    pdp = []
    
    with torch.no_grad():
        for bin_idx in range(num_bins):
            # Use bin center
            bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
            
            # Get all node indices (or subsample)
            all_node_idx = np.arange(dataset.x.shape[0])
            if max_bin_size is not None:
                all_node_idx = np.random.choice(
                    all_node_idx,
                    size=min(max_bin_size, len(all_node_idx)),
                    replace=False,
                )
            
            data = dataset.clone()
            data.to(device)
            
            # Sample k random target nodes
            sample_nodes = np.random.choice(dataset.x.shape[0], size=k, replace=False)
            
            # Create edges: all source nodes to all sample nodes (Cartesian product)
            edge_label_index = torch.stack([
                torch.tensor(all_node_idx, dtype=torch.long).repeat_interleave(k),
                torch.tensor(sample_nodes, dtype=torch.long).repeat(len(all_node_idx))
            ]).to(device)
            
            # Set ALL selected nodes' feature to bin center
            data.x[all_node_idx, feature_index] = torch.tensor(
                bin_center, device=device, dtype=torch.float32
            )
            
            # Get predictions
            logits = model(data.x, data.edge_index, edge_label_index).cpu().numpy()
            predictions = 1 / (1 + np.exp(-logits))
            
            # PDP value is average prediction
            pdp.append(float(np.mean(predictions)))
    
    end = time.time()
    return pdp, end - start

def ALE_exact(
    model,
    dataset,
    feature_index,
    num_bins=10,
    max_bin_size=None,
    k=256,
    device=torch.device("cuda"),
):
    start = time.time()
    feature_values = dataset.x[:, feature_index].cpu()
    bin_edges = np.linspace(
        feature_values.min() - 0.001, feature_values.max() + 0.001, num_bins + 1
    )
    bin_indices = np.digitize(feature_values.numpy(), bin_edges) - 1
    ale = []

    with torch.no_grad():
        for bin_idx in range(num_bins):
            bin_data_idx = np.where(bin_indices == bin_idx)[0]
            if len(bin_data_idx) == 0:
                ale.append(ale[-1] if ale else 0)
                continue

            # Subsample nodes in bin if max_bin_size specified
            if max_bin_size is not None:
                bin_data_idx = np.random.choice(
                    bin_data_idx,
                    size=min(max_bin_size, len(bin_data_idx)),
                    replace=False,
                )

            bin_diffs = []
            # Sample k random nodes to predict edges to
            sample_nodes = np.random.choice(dataset.x.shape[0], size=k, replace=True)
            for idx in bin_data_idx:
                data = dataset.clone()
                data.to(device)

                edge_label_index = torch.stack(
                    [
                        torch.full((k,), idx, dtype=torch.long),
                        torch.tensor(sample_nodes, dtype=torch.long),
                    ]
                ).to(device)

                data.x[idx, feature_index] = torch.tensor(
                    bin_edges[bin_idx], device=device, dtype=torch.float32
                )
                with torch.no_grad():
                    logits = (
                        model(data.x, data.edge_index, edge_label_index).cpu().numpy()
                    )
                    lower = 1 / (1 + np.exp(-logits))
                # lower = model.decode(model(data.x, data.edge_index), edge_label_index).view(-1).sigmoid()

                # Upper bound
                data.x[idx, feature_index] = torch.tensor(
                    bin_edges[bin_idx + 1], device=device, dtype=torch.float32
                )
                with torch.no_grad():
                    logits = (
                        model(data.x, data.edge_index, edge_label_index).cpu().numpy()
                    )
                    upper = 1 / (1 + np.exp(-logits))
                # upper = model.decode(model(data.x, data.edge_index), edge_label_index).view(-1).sigmoid()

                bin_diffs.extend((upper - lower).tolist())

            bin_effect = np.mean(bin_diffs)
            ale.append((ale[-1] if ale else 0) + bin_effect)

    end = time.time()
    return ale, bin_edges, end - start


def ALE_approximate(
    model,
    dataset,
    feature_index,
    num_bins=10,
    max_bin_size=None,
    k=256,
    device=torch.device("cuda"),
):
    start = time.time()
    feature_values = dataset.x[:, feature_index].cpu()
    bin_edges = np.linspace(
        feature_values.min() - 0.001, feature_values.max() + 0.001, num_bins + 1
    )
    bin_indices = np.digitize(feature_values.numpy(), bin_edges) - 1
    ale = []

    with torch.no_grad():
        for bin_idx in range(num_bins):
            bin_data_idx = np.where(bin_indices == bin_idx)[0]
            if len(bin_data_idx) == 0:
                ale.append(ale[-1] if ale else 0)
                continue

            # Subsample nodes in bin if max_bin_size specified
            if max_bin_size is not None:
                bin_data_idx = np.random.choice(
                    bin_data_idx,
                    size=min(max_bin_size, len(bin_data_idx)),
                    replace=False,
                )

            data = dataset.clone()
            data.to(device)

            # Sample k random nodes to predict edges to
            sample_nodes = np.random.choice(dataset.x.shape[0], size=k, replace=True)

            # Create edges: all bin nodes to all sample nodes (Cartesian product)
            edge_label_index = torch.stack(
                [
                    torch.tensor(bin_data_idx, dtype=torch.long).repeat_interleave(k),
                    torch.tensor(sample_nodes, dtype=torch.long).repeat(
                        len(bin_data_idx)
                    ),
                ]
            ).to(device)

            # Lower bound

            data.x[bin_data_idx, feature_index] = torch.tensor(
                bin_edges[bin_idx], device=device, dtype=torch.float32
            )
            with torch.no_grad():
                logits = model(data.x, data.edge_index, edge_label_index).cpu().numpy()
                lower = 1 / (1 + np.exp(-logits))
            # lower = model.decode(model(data.x, data.edge_index), edge_label_index).view(-1).sigmoid()

            # Upper bound
            data.x[bin_data_idx, feature_index] = torch.tensor(
                bin_edges[bin_idx + 1], device=device, dtype=torch.float32
            )
            with torch.no_grad():
                logits = model(data.x, data.edge_index, edge_label_index).cpu().numpy()
                upper = 1 / (1 + np.exp(-logits))
            # upper = model.decode(model(data.x, data.edge_index), edge_label_index).view(-1).sigmoid()

            bin_effect = float(np.mean(upper - lower))
            ale.append((ale[-1] if ale else 0) + bin_effect)

    end = time.time()
    return ale, bin_edges, end - start


def loop(seed):
    device = "cuda"
    df_list = []
    E = 10000
    for sparsity in [0.001, 0.01, 0.1]:
        train_data = generate_directed_synthetic_graph(
            N=int(np.sqrt(E/sparsity)), D=6, feat_idx=0, k=6.0, threshold=0.0, edge_sparsity=sparsity, seed=seed

        )
        model = train_link_predictor(
            train_data, hidden=32, num_layers=2, epochs=100, lr=0.0005, device=device, batch_size=256
        )
        test_data = generate_directed_synthetic_graph(
            N=4096, D=6, feat_idx=0, k=6.0, threshold=0.0, edge_sparsity=sparsity, seed=seed
        )
        auc = evaluate_model_auc(model, test_data, device=device)
        print(f"Test AUC (trained on synthetic edges): {auc:.4f}")

        # Compute ALE-like curve for feat_idx
        feat_idx = test_data.feat_idx
        # build a grid around empirical percentiles of that feature
        xs = test_data.x[:, feat_idx].cpu().numpy()
        grid = np.linspace(np.percentile(xs, 1), np.percentile(xs, 99), 5)
        # grid, ale_vals = compute_ale_for_feature(model, test_data, feat_idx, grid, device=device,
        #                                        sample_node_indices=None)

        for k in range(4, 11):
            for max_bin_size in range(4, 11):
                for i in range(3):
                    ale_approximate, grid_a, t_approximate = ALE_approximate(#PDP
                        model, test_data, feat_idx, 5, 2**max_bin_size, 2**k
                    )
                    ale_exact, grid_e, t_exact = ALE_exact(
                        model, test_data, feat_idx, 5, 2**max_bin_size, 2**k
                    )
                
                    df_list.append(
                        pd.DataFrame(
                            {   
                                'i': i,
                                "feat": grid,
                                "k": 2**k,
                                "max_bin_size": 2**max_bin_size,
                                "explanation_approximate": ale_approximate,
                                "time_approximate": t_approximate,
                                "explanation_exact": ale_exact,
                                "time_exact": t_exact,
                                "groundtruth": grid/4+0.25,
                                'AUC': auc,
                                'sparsity': sparsity,
                            }
                        )
                    )
    return pd.concat(df_list)
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(grid, ale_approximate, marker="o")
    plt.plot(grid, ale_exact, marker="o")
    plt.title("Interventional ALE-like curve (predicted outgoing link prob)")
    plt.xlabel(f"Feature {feat_idx} value")
    plt.ylabel("Mean predicted outgoing link probability")
    plt.grid(True)
    plt.tight_layout()
    plt.show()  # savefig('ale_curve.png')
    print("Saved ale_curve.png")


if __name__ == "__main__":
    
    for i in range(3):
        df = loop(i)
        df['seed'] = i
        df.to_csv(os.path.join("data", "synthetic2", f"ALE_GAT_n_layers2_hidden_size64_lin5_big.csv"), mode='a', header=False)#SAGEConv
