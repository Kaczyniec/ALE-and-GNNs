import os.path as osp
import time
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np
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
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from torch_geometric.utils import negative_sampling
import sys
if "C:\\Users\\ppaul\\Documents" not in sys.path:
    sys.path.append("C:\\Users\\ppaul\\Documents")
from influence_on_ideas.utils.preprocess_data import graph_data

CONFIGS = [{'hidden_channels': 64, 
           'lr': 0.01, 
           #'weight_decay': 5e-4, 
           #'epochs': 100, 
           #'batch_size': 64, 
           'n_layers': 2, 
           'model_type': 'GAT'
           },
           {'hidden_channels': 64, 
           'lr': 0.01, 
           #'weight_decay': 5e-4, 
           #'epochs': 100, 
           #'batch_size': 64, 
           'n_layers': 2, 
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
        # cosine similarity

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
        # standard torch mechanics here
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch_size
        total_examples += batch_size
        logging.debug(f'Loss: {loss.item()}')
        scheduler.step()
    if i % 100 == 0:
        torch.save(
            model.state_dict(), "models/citations/"+CONFIGS['model_type']+',n_layers'+str(CONFIGS['n_layers'])+',hidden_size'+str(CONFIGS['hidden_size'])
        )
    return total_loss, total_examples


@torch.no_grad()
def test(model, loader):

    """
    Evalutes the model on the test set.
    :param loader: the batch loader
    :return: a score
    """
    model.eval()
    f1scores = []
    accuracies = []
    threshold = torch.tensor([0.7]).to(device)
    for batch in tqdm(loader):
        batch.to(device)
        z = model.forward(batch.x, batch.edge_index)

        out = model.decode(z, batch.edge_label_index).view(-1).sigmoid()

        pred = (out > threshold).float() * 1
        pred = pred.detach().cpu().numpy()
        edge_label = batch.edge_label.detach().cpu().numpy()
        f1score = f1_score(edge_label, pred)
        f1scores.append(f1score)
        fpr, tpr, thresholds = roc_curve(np.ones(batch.edge_index.size(1)), out.cpu().numpy())
        logging.debug(f'Batch size: {batch.size(0)}, F1 score: {f1score}')
        logging.debug(f'fpr: {fpr}, tpr: {tpr}, thresholds: {thresholds}')
    return np.average(f1scores), np.average(accuracies)

if __name__ == '__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #edges = pd.read_csv('data/twitch/large_twitch_edges.csv')
    #node_features = pd.read_csv('data/twitch/large_twitch_features.csv') 
    edges = pd.read_csv('data/citations/edge.csv', index_col=0)
    node_features = pd.read_csv('data/citations/node_features.csv', index_col=0)  
    train_loader, test_loader, train_data, test_data = graph_data(edges, node_features)
    model = Model(in_channels= np.shape(train_data.x)[1], hidden_channels=64, model_type=CONFIGS['model_type'], n_layers=CONFIGS['n_layers']).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    loss_values = []
    for epoch in range(1, CONFIGS['epochs']):
        logging.info(f'Starting epoch {epoch}')
        loss = train(train_loader, device, optimizer, model, scheduler)
        f1 = test(test_loader, model)
        loss_values.append(loss)  # Store the loss value
        logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, F1: {f1:.5f}')
        torch.save(model.state_dict(), 'models/citations/'+str())
