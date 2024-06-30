import random
import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm
import torch_geometric.transforms as T
import sys
if "/home/pkaczynska/repositories" not in sys.path:
    sys.path.append("/home/pkaczynska/repositories")
from influence_on_ideas.utils.preprocess_data import graph_data
from influence_on_ideas.models.twitch_gnn import Model
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import k_hop_subgraph
import os
import argparse


def graph_pdp_exact(dataset, model, column, values, max_bin_size=256, k=256, device='cpu'):
    '''function to plot the partial dependence plot
    data: ??
    model: the pyg model for link prediction
    column: the column index
    values: list of values to check the partial dependence for
    Example: How much probable is on average to cite a paper if it is coauthored solely by Big tech authors than having a mixed affiliation?
    '''
    start = time.time()
    probabilities = {}
    if max_bin_size is not None:
      random_papers = np.random.choice(range(dataset.x.shape[0]), size=max_bin_size, replace=False).tolist()
    else:
      random_papers = np.array(range(dataset.x.shape[0])).tolist()
    
    model.eval()
    with torch.no_grad():
      for value in tqdm(values):
        
        for paper in random_papers:
            data = dataset.clone()
          
            data.x[paper, column] = torch.tensor(value)
            unique = np.random.choice(range(dataset.x.shape[0]), size=k, replace=False)
            subset, edge_index, mapping, edge_mask = k_hop_subgraph([paper]+list(unique), 2, data.edge_index)

            data.edge_index = edge_index
            edge_label_index = torch.cat((torch.Tensor(unique).unsqueeze(-1), int(paper)*torch.ones(unique.shape).unsqueeze(-1)),dim=1).type(torch.LongTensor).to(device)
            data.to(device)
            preds = []

            encode = model(data.x, data.edge_index)
            out = model.decode(encode, edge_label_index)
            preds.append(out)

        probabilities[value] = np.mean(torch.cat(preds).detach().cpu().numpy())
    end = time.time()
    #plt.plot(values, probabilities.values())
    #plt.xlabel(column)
    #plt.ylabel('Mean probability of being cited')
    return probabilities, end-start

def graph_pdp_approximate(dataset, model, column, values, max_bin_size=256, k=256, device='cpu'):
    '''function to plot the partial dependence plot
    data: ??
    model: the pyg model for link prediction
    column: the column index
    values: list of values to check the partial dependence for
    Example: How much probable is on average to cite a paper if it is coauthored solely by Big tech authors than having a mixed affiliation?
    '''
    start = time.time()
    probabilities = {}
    if max_bin_size is not None:
      random_papers = np.random.choice(range(dataset.x.shape[0]), size=max_bin_size, replace=False)
    else:
      random_papers = np.array(range(dataset.x.shape[0]))
    for value in tqdm(values):
      probabilities[value] = []
      data = dataset.clone()
      data.x[random_papers, column] = torch.tensor(value).float()
      unique = np.random.choice(range(dataset.x.shape[0]), size=k, replace=False)
      subset, edge_index, mapping, edge_mask = k_hop_subgraph(list(random_papers)+list(unique), 2, data.edge_index)

      data.edge_index = edge_index
      edge_label_index = torch.cat((torch.Tensor(unique).int().repeat(random_papers.shape[0]).unsqueeze(-1), torch.Tensor(random_papers).int().repeat_interleave(unique.shape[0]).unsqueeze(-1)),dim=1).to(device)

      model.eval()
      with torch.no_grad():
        z = model.forward(data.x, data.edge_index)
        out = model.decode(z, edge_label_index).view(-1).sigmoid()
        probabilities[value].append(np.mean(out.detach().cpu().numpy()))
    end = time.time()

    return probabilities, end-start

if __name__ == '__main__':
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("Running on: ", device)
  parser = argparse.ArgumentParser(description='Graph Neural Network Training Script')
  parser.add_argument('--edges_path', type=str, required=True, help='Path to the edges CSV file')
  parser.add_argument('--node_features_path', type=str, required=True, help='Path to the node features CSV file')
  parser.add_argument('--name', type=str, required=True, help='Dataset name')
  parser.add_argument('--model_type', type=str, default='GCN', help='Architecture')
  parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
  parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
  parser.add_argument('--column', type=int, default=2, help='Number of column to explain')
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
  model_path = os.path.join("models", args.name, args.model_type+",n_layers"+str(args.n_layers)+",hidden_size"+str(args.hidden_dim))
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
  #train_data.to(device)
  test_data.to(device)
  results = pd.DataFrame(columns=['idx', 'k', 'max_bin_size', 'explanation_exact', 'time_exact', 'explanation_approximate', 'time_approximate'])
  for k in range(4, 11):
    for max_bin_size in range(4, 11):
      print(k, max_bin_size)
      for i in range(5):
          ale_exact, t_exact = graph_pdp_exact(test_data,model, args.column, [0, 0.25, 0.5, 0.75, 1], 2**max_bin_size, 2**k, device)
          ale_approximate, t_approximate = graph_pdp_approximate(test_data,model, args.column, [0, 0.25, 0.5, 0.75, 1], 2**max_bin_size, 2**k, device)

          results = pd.concat([results, pd.DataFrame({'idx': i, 'k': 2**k, 'max_bin_size': 2**max_bin_size, 'explanation_exact': ale_exact, 'time_exact': t_exact, 'explanation_approximate': ale_approximate, 'time_approximate': t_approximate})])
          results.to_csv(os.path.join("data", args.name, f"PDP_{args.model_type}_n_layers{args.n_layers}_hidden_size{args.hidden_channels}.csv", mode='a', header=False))

