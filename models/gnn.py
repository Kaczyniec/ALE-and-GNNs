import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize, MultiLabelBinarizer

import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, GCNConv
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.models import GraphSAGE, GCN, GAT
from torch_geometric.data import InMemoryDataset
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from torch_geometric.utils import negative_sampling
import sys
if "/home/pkaczynska/repositories" not in sys.path:
    sys.path.append("/home/pkaczynska/repositories")
from influence_on_ideas.utils.preprocess_data import graph_data
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
CONFIGS = [{'hidden_channels': 128, 
           'lr': 0.005, 
           #'weight_decay': 5e-4, 
           'epochs': 10, 
           #'batch_size': 64, 
           'n_layers': 3, 
           'model_type': 'GAT'
           },
           {'hidden_channels': 128, 
           'lr': 0.005, 
           #'weight_decay': 5e-4, 
           'epochs': 10, 
           #'batch_size': 64, 
           'n_layers': 3, 
           'model_type': 'GCN'
           },]
CONFIGS = CONFIGS[1]

class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, model_type, n_layers):
        super().__init__()
        if model_type == "GraphSAGE":
            self.model = GraphSAGE(in_channels, hidden_channels, n_layers)
        elif model_type == "GCN":
            self.model = GCN(in_channels, hidden_channels, n_layers)
        elif model_type == "GAT":
            self.model = GAT(in_channels, hidden_channels, n_layers)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.model(x, edge_index)

        return x

    def decode(self, z, edge_label_index):

        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()  # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t()  # get predicted edge_list


def train(train_loader, device, optimizer, model, scheduler):
    """
    Single epoch model training in batches.
    :return: total loss for the epoch
    """
    model.train()
    total_examples = total_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch.size()[0]
        z = model.forward(batch.x, batch.edge_index.type(torch.int64))
        neg_edge_index = negative_sampling(
            edge_index=batch.edge_index,
            num_nodes=batch.num_nodes,
            num_neg_samples=None,
            method="sparse",
        )
        edge_label_index = torch.cat(
            [batch.edge_label_index, neg_edge_index],
            dim=-1,
        ).to(device)
        edge_label = torch.cat(
            [
                torch.ones(batch.edge_label_index.size(1)),
                torch.zeros(neg_edge_index.size(1)),
            ],
            dim=0,
        ).to(device)
        out = model.decode(z, edge_label_index).view(-1).sigmoid()
        # loss = F.binary_cross_entropy_with_logits(out[:batch_size], edge_label[:batch_size])
        # print(out.get_device(), edge_label.get_device())
        loss = F.binary_cross_entropy_with_logits(out, edge_label)

        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch_size
        total_examples += batch_size
        logging.debug(f'Loss: {loss.item()}')
        scheduler.step()
    
    torch.save(
        model.state_dict(), "models/citations/"+CONFIGS['model_type']+',n_layers'+str(CONFIGS['n_layers'])+',hidden_size'+str(CONFIGS['hidden_channels'])
    )
    return total_loss, total_examples


@torch.no_grad()
def test(model, loader, device='cuda'):
    """
    Evaluates the model on the test set.
    :param model: the model to evaluate
    :param loader: the batch loader
    :param device: the device to run the evaluation on (default: 'cuda')
    :return: a dictionary containing average F1 score and ROC AUC score
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    threshold = torch.tensor([0.7]).to(device)
    
    for batch in tqdm(loader):
        batch.to(device)
        z = model(batch.x, batch.edge_index)  # Assuming model has __call__ method

        out = model.decode(z, batch.edge_label_index).view(-1).sigmoid()

        pred = (out > threshold).float()
        
        all_preds.append(pred.cpu().numpy())
        all_labels.append(batch.edge_label.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    avg_f1 = f1_score(all_labels, all_preds)
    avg_roc_auc = roc_auc_score(all_labels, all_preds)
    
    logging.info(f'F1 score: {avg_f1:.4f}, ROC AUC: {avg_roc_auc:.4f}')
    print(f'F1 score: {avg_f1:.4f}, ROC AUC: {avg_roc_auc:.4f}')
    return {'average_f1_score': avg_f1, 'average_roc_auc': avg_roc_auc}

if __name__ == '__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Training on: ", device)
    #edges = pd.read_csv('data/twitch/large_twitch_edges.csv')
    #node_features = pd.read_csv('data/twitch/large_twitch_features.csv') 
    train_data = torch.load('data/citations_train_data')
    edges = pd.read_parquet('data/citations/edge.parquet')
    node_features = pd.read_parquet('data/citations/node_features.parquet')  
    train_loader, test_loader, train_data, test_data = graph_data(edges, node_features)
    #torch.save(train_data, 'data/citations/train_data')
    #torch.save(test_data, 'data/citations/test_data')

    model = Model(in_channels= np.shape(train_data.x)[1], hidden_channels=CONFIGS['hidden_channels'], model_type=CONFIGS['model_type'], n_layers=CONFIGS['n_layers']).to(device)
    # Check if the file exists
    model_path = "models/citations/"+CONFIGS['model_type']+',n_layers'+str(CONFIGS['n_layers'])+',hidden_size'+str(CONFIGS['hidden_channels'])
    if os.path.isfile(model_path):
        # Load the weights into the model
        model.load_state_dict(torch.load(model_path))
        print("Model weights loaded successfully.")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=CONFIGS['lr'])
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    loss_values = []
    for epoch in range(CONFIGS['epochs']):
        loss = train(train_loader, device, optimizer, model, scheduler)
        
        metrics = test(model, test_loader)
        loss_values.append(loss)