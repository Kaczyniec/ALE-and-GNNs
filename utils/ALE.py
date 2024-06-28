import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm
import sys
import argparse
import os

from torch_geometric.utils import k_hop_subgraph
#if "/home/pkaczynska/repositories" not in sys.path:
#    sys.path.append("/home/pkaczynska/repositories")
path_to_add = "C:/Users/ppaul/Documents"

if path_to_add not in sys.path:
    # Add the path to sys.path
    sys.path.append(path_to_add)
from influence_on_ideas.utils.preprocess_data import graph_data
from influence_on_ideas.models.twitch_gnn import Model


#Change: the prediction is between node with change value and the rest of the dataset!!!
def accumulated_local_effects_exact(model, dataset, feature_index, num_bins=10, max_bin_size = None, k=256, device=torch.device('cuda')):
    start = time.time()
    # Step 1: Divide the range of values of the selected variable into bins
    feature_values = dataset.x[:, feature_index].cpu()

    bin_edges = np.linspace(feature_values.min()-0.001, feature_values.max()+0.001, num_bins + 1)

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
          if bin_data_idx.shape[0]==0 and len(ale)>0:
            ale.append(ale[-1])
            continue

          if max_bin_size!=None:
            bin_data_idx_subset = np.random.choice(bin_data_idx, size=min(max_bin_size, bin_data_idx.shape[0]), replace=False)
          else:
            bin_data_idx_subset = bin_data_idx
          for idx in bin_data_idx_subset:
            data = dataset.clone()
            model.eval()

            # Step 3: Calculate model predictions for lower and upper end of the section
            unique = np.random.choice(list(nodes), size=k)
            
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(list(unique)+[idx], 2, data.edge_index, relabel_nodes=True)
            
            edge_label_index = torch.cat((torch.Tensor(mapping[:-1]).int().unsqueeze(-1), mapping[-1]*torch.ones(k).int().unsqueeze(-1)),dim=1)
            data.edge_index = edge_index
            
            data.to(device)
            data.x[idx, feature_index] = torch.tensor(bin_edges[bin_idx]).float()
            lower_encode = model(data.x[subset], data.edge_index)
            lower = model.decode(lower_encode, edge_label_index)
            data.x[idx, feature_index] = torch.tensor(bin_edges[bin_idx + 1]).float()
            upper_encode = model(data.x[subset], data.edge_index)
            upper = model.decode(upper_encode, edge_label_index)
      
            # Step 4: Subtract the above values. Average across all data points in the bin

            bin_ale = float(torch.mean((upper-lower)).detach())

          # Step 5: Calculate the cumulative sum of the averaged differences
          bin_ale = np.mean(bin_ale)
          if ale:
              ale.append(bin_ale + ale[-1])
          else:
              ale.append(bin_ale)

      end = time.time()
      return ale, end-start

    
def accumulated_local_effects_approximate(model, dataset, feature_index, num_bins=10, max_bin_size = None, k=256, device=torch.device('cuda')):
    start = time.time()
    # Step 1: Divide the range of values of the selected variable into bins
    feature_values = dataset.x[:, feature_index].cpu()

    bin_edges = np.linspace(feature_values.min()-0.0001, feature_values.max()+0.0001, num_bins + 1)

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
          data = dataset.clone()
          bin_data_idx = np.where(bin_indices == bin_idx)[0]
          if bin_data_idx.shape[0]==0 and len(ale)>0:
            ale.append(ale[-1])
            continue

          if max_bin_size!=None:
            bin_data_idx_subset = np.random.choice(bin_data_idx, size=min(max_bin_size, bin_data_idx.shape[0]), replace=False)
          else:
            bin_data_idx_subset = bin_data_idx

          model.eval()

          # Step 3: Calculate model predictions for lower and upper end of the section
          unique = np.random.choice(list(nodes), size=k)

          #create a tensor with edges to predict: for every node from bin_data_idx_subset check connections to every node in unique
          subset, edge_index, mapping, edge_mask = k_hop_subgraph(list(bin_data_idx_subset)+list(unique), 2, dataset.edge_index, relabel_nodes=True)
          
          data.edge_index = edge_index

          unique = torch.Tensor(unique).int()
          edge_label_index = torch.cat(
              (torch.Tensor(mapping[:len(bin_data_idx_subset)]).int().repeat_interleave(unique.shape[0]).unsqueeze(-1),
               mapping[len(bin_data_idx_subset):].repeat(bin_data_idx_subset.shape).int().unsqueeze(-1)), axis=1)
          data.x[bin_data_idx_subset, feature_index] = torch.tensor(bin_edges[bin_idx]).float()
          data.to(device)
          lower_encode = model(data.x[subset], data.edge_index)
          lower = model.decode(lower_encode, edge_label_index).view(-1).sigmoid()

          dataset.x[bin_data_idx_subset, feature_index] = torch.tensor(bin_edges[bin_idx + 1]).float()

          upper_encode = model(data.x[subset], data.edge_index)
          upper = model.decode(upper_encode, edge_label_index).view(-1).sigmoid()

            #preds.append(upper-lower)

          # Step 4: Subtract the above values. Average across all data points in the bin

          bin_ale.append(float(torch.mean(upper-lower).detach()))

          if ale:
              ale.append(bin_ale + ale[-1])
          else:
              ale.append(bin_ale)

      end = time.time()
      return ale, end-start
    
    

if __name__ == '__main__':
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("Running on: ", device)
  EDGES_PATH="data/CD1-E_no2/CD1-E-no2_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"
  NODE_FEATURES_PATH="data/CD1-E_no2/CD1-E-no2_iso3um_stitched_segmentation_bulge_size_3.0_nodes.csv"
  DATASET_NAME="CD1-E_no2"
  parser = argparse.ArgumentParser(description='Graph Neural Network Explaining Script')
  parser.add_argument('--edges_path', type=str, default=EDGES_PATH, help='Path to the edges CSV file')
  parser.add_argument('--node_features_path', type=str, default=NODE_FEATURES_PATH, help='Path to the node features CSV file')
  parser.add_argument('--name', type=str, default=DATASET_NAME, help='Dataset name')
  parser.add_argument('--model_type', type=str, default='GCN', help='Architecture')
  parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
  parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
  parser.add_argument('--column', type=int, default=3, help='Which column explain?')
  args = parser.parse_args()


  #edges_path = 'data/citations/edge.parquet'
  #node_features_path = 'data/citations/node_features.parquet'
  #data_path = 'data/citations/'
  edges = pd.read_csv(args.edges_path, sep=';')[['node1id', 'node2id']]
  node_features = pd.read_csv(args.node_features_path, sep=';')[['pos_x', 'pos_y',  'pos_z', 'isAtSampleBorder']]
  #edges = pd.read_parquet(args.edges_path)
  #node_features = pd.read_parquet(args.node_features_path)
  
  train_loader, test_loader, train_data, test_data = graph_data(edges, node_features, "data/"+args.name)
  #model_path = f"models/citations/{config['model_type']},n_layers{config['n_layers']},hidden_size{config['hidden_channels']}"
  model_path = os.path.join("models", args.name, args.model_type+",n_layers"+str(args.n_layers)+",hidden_size"+str(args.hidden_dim))+"lr1e-06.pth"
  print(model_path)
  # Initialize the model
  model = Model(
      in_channels=np.shape(train_data.x)[1], 
      hidden_channels=args.hidden_dim, 
      model_type=args.model_type, 
      n_layers=args.n_layers
  ).to(device)

  # Check if the model weights file exists
  if os.path.isfile(model_path):
      # Load the weights into the model
      model.load_state_dict(torch.load(model_path))
      print("Model weights loaded successfully.")
  else:
      print("Model weights file does not exist. Initializing model with random weights.")
  
  
  results = pd.DataFrame(columns=['idx', 'k', 'max_bin_size', 'explanation_exact', 'time_exact', 'explanation_approximate', 'time_approximate'])
  for k in range(4, 11):
    for max_bin_size in range(4, 11):
      print(k, max_bin_size)
      for i in range(5):
          ale_exact, t_exact = accumulated_local_effects_exact(model,test_data, args.column, 5, 2**max_bin_size, 2**k, device)
          ale_approximate, t_approximate = accumulated_local_effects_approximate(model,test_data, args.column, 5, 2**max_bin_size, 2**k, device)
          results = pd.concat([results, pd.DataFrame({'idx': i, 'k': 2**k, 'max_bin_size': 2**max_bin_size, 'explanation_exact': ale_exact, 'time_exact': t_exact, 'explanation_approximate': ale_approximate, 'time_approximate': t_approximate})])
          results.to_csv(os.path.join("data", args.name, f"ALE_{args.model_type}_n_layers{args.n_layers}_hidden_size{args.hidden_dim}.csv"), mode='a', header=False)
