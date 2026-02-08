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
from sklearn.metrics import roc_auc_score, precision_score, recall_score
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
from utils.synthetic import PDP_approximate, PDP_exact
from torch_geometric.utils import k_hop_subgraph
from torch.optim.lr_scheduler import ExponentialLR
import torch_geometric.transforms as T
from utils.preprocess_data import graph_data
from models.gnn_batchnorm import Model, train, test
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce

from utils.ALE import (
    accumulated_local_effects_approximate,
    accumulated_local_effects_exact,
)
import os

from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np
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

    X = np.random.normal(0, 1, size=(N, D)).astype(np.float32)
    # Optionally make the signal feature more spread to see effects
    X[:, feat_idx] = np.random.uniform(-1, 1, size=(N,))

    # Build edges
    sources = []
    targets = []
    labels = []
    nsources = []
    ntargets = []
    num_candidates_per_node = int(edge_sparsity * N)
    for u in range(N):
        # choose candidate targets without replacement (excluding self)
        candidates = list(range(N))
        candidates.remove(u)
        sampled = np.random.choice(
            candidates, size=num_candidates_per_node, replace=False
        )
        p = sigmoid(k * (X[u, feat_idx]))# - X[sampled, feat_idx]#(X[u, feat_idx]-X[sampled, feat_idx]+2)/4#np.exp(X[u, feat_idx]-X[sampled, feat_idx])/np.exp(2)-np.exp(-1)#
        for v in sampled:#v, p in zip(sampled,prob):
            if np.random.rand() < p:
                # create directed edge u -> v
                sources.append(u)
                targets.append(v)

            else:
                # negative sample (explicitly include with label 0)
                nsources.append(u)
                ntargets.append(v)

    print("Nodes and edegs:", N, len(sources))
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_label_index = torch.cat((edge_index, torch.tensor([nsources, ntargets], dtype=torch.long)), dim=1)
    x = torch.tensor(X, dtype=torch.float)
    edge_label = torch.cat((torch.ones(len(sources)), torch.zeros(len(nsources))))
    #print(edge_label.float().mean())
    data = Data(x=x, edge_index=edge_index, edge_label=edge_label, edge_label_index=edge_label_index)
    data.N = N
    data.D = D
    data.feat_idx = feat_idx
    data.gen_k = k
    data.gen_threshold = threshold
    return data

def train_link_predictor(
    data, hidden=64, num_layers=2, epochs=30, batch_size=2048, lr=5e-3, device="cpu"
):
    device = torch.device(device)
    model = Model(
            in_channels=data.D, 
            hidden_channels=hidden,
            model_type='GraphSAGE',
            n_layers=num_layers,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #x = data.x.to(device)
    #edge_index = data.edge_index.to(device)
    #edge_label = data.edge_label.to(device)

    # We'll train in randomized mini-batches over the provided edges (which include both 1 and 0 labels)
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=[10] * 2,
        batch_size=batch_size,
        edge_label_index=data.edge_label_index,
        edge_label=data.edge_label,
        #neg_sampling="binary",
        #neg_sampling_ratio=5,
        #time_attr=, 
    )

    for ep in range(epochs):
        model.train()
        total_examples = total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            batch_size = batch.size()[0]

            z = model.forward(batch.x, batch.edge_index.type(torch.int64))

            out = model.decode(z, batch.edge_label_index).view(-1)#.sigmoid()
            #print("Mean value of training label:", batch.edge_label.float().mean())
            loss = F.binary_cross_entropy_with_logits(out, batch.edge_label.float(), reduction='mean')

            loss.backward()
            optimizer.step()
            total_loss += float(loss) * batch_size
            total_examples += batch_size

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{epochs} loss={np.mean(total_loss/total_examples):.4f}")

    return model


# ----------------------------- Evaluation --------------------------------


def evaluate_model_auc(model, data, device="cpu"):
    model.eval()
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[10] * 2,
        batch_size=32,
        edge_label_index=data.edge_label_index,
        edge_label=data.edge_label,
        #neg_sampling_ratio=1.0,
        #neg_sampling="binary",
        #neg_sampling_ratio=1.0,
    )
    all_preds = []
    all_labels = []

    for batch in loader:
        batch.to(device)
        z = model(batch.x, batch.edge_index)  # Assuming model has __call__ method

        out = model.decode(z, batch.edge_label_index).view(-1).sigmoid()
    

        all_preds.append(out)
        all_labels.append(batch.edge_label)

    all_preds = torch.concat(all_preds).detach().cpu().numpy()
    all_labels = torch.concat(all_labels).detach().cpu().numpy()
    print(all_labels.mean(), all_preds.mean())

    avg_roc_auc = roc_auc_score(all_labels, all_preds)
    threshold=0.5
    precision = precision_score(all_labels, all_preds > threshold, average='binary')
    recall = recall_score(all_labels, all_preds > threshold, average='binary')
    print("precission ", precision, "recall", recall)
    #avg_f1 = f1_score(all_labels, (all_preds > threshold).float())
    #acc = (all_preds==all_labels).mean()
    return avg_roc_auc

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
                z = model.forward(data.x, data.edge_index.type(torch.int64))

                out = model.decode(z, edge_label_index).view(-1).sigmoid().cpu().numpy()

                
                # Store average prediction for this source node
                bin_predictions.append(np.mean(out))
            
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
            
            z = model.forward(data.x, data.edge_index.type(torch.int64))

            out = model.decode(z, edge_label_index).view(-1).sigmoid().cpu().numpy()


            # PDP value is average prediction
            pdp.append(float(np.mean(out)))
    
    end = time.time()
    return pdp, end - start
def loop(seed):
    device = "cuda"
    df_list = []
    E = 10000
    for sparsity in [0.005]:#[0.001, 0.01, 0.1]:
        train_data = generate_directed_synthetic_graph(
            N=2*int(np.sqrt(E/sparsity)), D=6, feat_idx=0, k=6.0, threshold=0.0, edge_sparsity=sparsity, seed=seed
        )
        model = train_link_predictor(
            train_data, hidden=128, num_layers=2, epochs=200, lr=0.00001, device=device, batch_size=512
        )
        test_data = generate_directed_synthetic_graph(
            N=1024, D=6, feat_idx=0, k=6.0, threshold=0.0, edge_sparsity=sparsity, seed=seed
        )
        auc = evaluate_model_auc(model, test_data, device=device)
        print(f"Test AUC (trained on synthetic edges): {auc:.4f}")

        # Compute ALE-like curve for feat_idx
        feat_idx = test_data.feat_idx
        # build a grid around empirical percentiles of that feature
        xs = test_data.x[:, feat_idx].cpu().numpy()
        grid = np.linspace(np.percentile(xs, 1), np.percentile(xs, 99), 21)
        # grid, ale_vals = compute_ale_for_feature(model, test_data, feat_idx, grid, device=device,
        #                                        sample_node_indices=None)

        for k in range(4, 10):
            for max_bin_size in range(4, 10):
                ale_approximate, t_approximate = PDP_approximate(#accumulated_local_effects_approximate(#PDP
                    model, test_data, feat_idx, 21, 2**max_bin_size, 2**k
                )
                ale_exact, t_exact = PDP_exact(#accumulated_local_effects_exact(
                    model, test_data, feat_idx, 21, 2**max_bin_size, 2**k
                )
                for i in range(3):
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
                                "groundtruth": sigmoid(6*grid),
                                'AUC': auc,
                                'sparsity': sparsity,
                            }
                        )
                    )
    return pd.concat(df_list)



if __name__ == "__main__":
    
    for i in range(2,5):
        df = loop(i)
        df['seed'] = i
        df.to_csv(os.path.join("data", "synthetic1", f"PDP_SAGE_n_layers2_hidden_size64.csv"), mode='a', header=False)#SAGEConv
