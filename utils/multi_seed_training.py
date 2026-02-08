import torch
import pandas as pd
import numpy as np
import os
import logging
import scipy.stats as stats
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocess_data import graph_data
from models.gnn_batchnorm import Model, train, test


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_single_seed(config, edges, node_features, data_path, device, seed):
    """Train a single model with given seed and return evaluation metrics"""
    print(f"Training with seed: {seed}")
    set_seed(seed)
    
    # Load data
    train_loader, test_loader, train_data, test_data, val_data, val_loader = graph_data(
        edges, node_features, data_path
    )
    
    # Initialize model
    model = Model(
        in_channels=train_data.x.shape[1],
        hidden_channels=config["hidden_channels"],
        model_type=config["model_type"],
        n_layers=config["n_layers"],
    ).to(device)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"])
    
    # Training loop
    model.train()
    for epoch in range(config["epochs"]):
        print(f"  Epoch {epoch+1}/{config['epochs']}")
        total_loss = 0
        total_examples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            batch_size = batch.size(0) if hasattr(batch, 'size') else len(batch.edge_label_index[0])

            z = model.forward(batch.x, batch.edge_index.type(torch.int64))
            
            # Generate negative samples
            neg_edge_index = negative_sampling(
                edge_index=batch.edge_index,
                num_nodes=batch.num_nodes,
                num_neg_samples=None,
                method="sparse",
            )

            edge_label_index = torch.cat(
                [batch.edge_label_index, neg_edge_index], dim=-1
            )
            edge_label = torch.cat(
                [
                    torch.ones(batch.edge_label_index.size(1)),
                    torch.zeros(neg_edge_index.size(1)),
                ],
                dim=0,
            ).to(device)

            out = model.decode(z, edge_label_index).view(-1)#.sigmoid()
            loss = F.binary_cross_entropy_with_logits(out, edge_label, reduce="mean")

            loss.backward()
            optimizer.step()
            total_loss += float(loss) * batch_size
            total_examples += batch_size

        print(f"  Epoch {epoch+1} Loss: {total_loss/total_examples:.4f}")
    
    # Evaluation
    print("  Evaluating...")
    val_metrics = test(model, val_loader, device)
    test_metrics = test(model, test_loader, device)
    
    
    return {
        'seed': seed,
        'val_f1': float(val_metrics['average_f1_score']),
        'val_auc': float(val_metrics['average_roc_auc']),
        'test_f1': float(test_metrics['average_f1_score']),
        'test_auc': float(test_metrics['average_roc_auc']),
    }


def calculate_confidence_intervals(results, metric_name, confidence=0.95):
    """Calculate confidence intervals for given metric"""
    values = [r[metric_name] for r in results]
    mean = np.mean(values)
    sem = stats.sem(values)  # Standard error of the mean
    h = sem * stats.t.ppf((1 + confidence) / 2., len(values)-1)
    
    return {
        'mean': mean,
        'std': np.std(values),
        'ci_lower': mean - h,
        'ci_upper': mean + h,
        'values': values
    }


def multi_seed_training_evaluation(config, edges, node_features, data_path, device, seeds):
    """Train multiple models with different seeds and calculate confidence intervals"""
    results = []
    
    for seed in seeds:

        result = train_single_seed(config, edges, node_features, data_path, device, seed)
        results.append(result)

    
    if not results:
        raise RuntimeError("No successful training runs completed")
    
    # Calculate confidence intervals
    metrics_ci = {}
    for metric in ['val_f1', 'val_auc', 'test_f1', 'test_auc']:
        metrics_ci[metric] = calculate_confidence_intervals(results, metric)
    
    # Print results
    print("\n" + "="*60)
    print("MULTI-SEED TRAINING RESULTS")
    print("="*60)
    
    for metric_name, ci_data in metrics_ci.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Mean: {ci_data['mean']:.4f}")
        print(f"  Std:  {ci_data['std']:.4f}")
        print(f"  95% CI: [{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}]")
        print(f"  Values: {[f'{v:.4f}' for v in ci_data['values']]}")
    
    return results, metrics_ci


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed GNN Training and Evaluation")
    parser.add_argument(
        "--edges_path", type=str, help="Path to the edges CSV file"
    )
    parser.add_argument(
        "--node_features_path",
        type=str,
        help="Path to the node features CSV file",
    )
    parser.add_argument(
        "--name",
        type=str,
    )
    parser.add_argument("--model_type", type=str, default="GCN", choices=["GCN", "GAT"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.000001)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seeds", nargs='+', type=int, default=[42, 123, 456, 789, 101])
    parser.add_argument("--dataset", type=str, help="Dataset name")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Configuration
    config = {
        "hidden_channels": args.hidden_dim,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_layers": args.n_layers,
        "model_type": args.model_type,
    }
    
    print(f"Configuration: {config}")
    print(f"Seeds: {args.seeds}")
    
    # Load data
    if args.edges_path.endswith('csv'):
        edges = pd.read_csv(args.edges_path, index_col=0, sep=";")[["node1id", "node2id"]]
        node_features = pd.read_csv(args.node_features_path, sep=";")[["pos_x", "pos_y", "pos_z", "isAtSampleBorder"]]
    else:
        edges = pd.read_parquet(args.edges_path)
        node_features = pd.read_parquet(args.node_features_path)
        
    # Run multi-seed training
    results, metrics_ci = multi_seed_training_evaluation(
        config, edges, node_features, "data/" + args.name, device, args.seeds
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"multi_seed_results_{args.model_type}_layers{args.n_layers}_hidden{args.hidden_dim}_{timestamp}.json"
    
    output_data = {
        'config': config,
        'seeds': args.seeds,
        'individual_results': results,
        'confidence_intervals': metrics_ci,
        'timestamp': timestamp
    }
    
    os.makedirs("results", exist_ok=True)
    with open(f"results/{results_file}", 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: results/{results_file}")