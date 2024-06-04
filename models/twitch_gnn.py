import os.path as osp
import time
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np
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

PATH = "drive/MyDrive/ALE_GNN/model_GAT_16.04"

node_features = pd.read_csv(
    "/content/drive/MyDrive/ALE_GNN/Twitch/large_twitch_features.csv"
)
edges = pd.read_csv("/content/drive/MyDrive/ALE_GNN/Twitch/large_twitch_edges.csv")

data = Data(
    x=torch.from_numpy(
        node_features[
            ["views", "mature", "life_time", "dead_account", "affiliate"]
        ].values
    ).float(),
    edge_index=torch.Tensor(edges.values).T.int(),
    y=node_features.affiliate,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.RandomLinkSplit(
    num_val=0.4,
    num_test=0.1,
    # disjoint_train_ratio=0.3,
    # neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
)

train_data, val_data, test_data = transform(data)
train_data, val_data, test_data = (
    train_data.to(device),
    val_data.to(device),
    test_data.to(device),
)

# Define seed edges:
edge_label_index = train_data.edge_label_index
edge_label = train_data.edge_label


train_loader = LinkNeighborLoader(
    train_data,
    num_neighbors=[10] * 2,
    batch_size=1024,
    edge_label_index=edge_label_index,
    edge_label=edge_label,
)
test_loader = LinkNeighborLoader(
    test_data,
    num_neighbors=[10] * 2,
    batch_size=512,
    edge_label_index=test_data.edge_label_index,
    edge_label=test_data.edge_label,
)


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


from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.utils import negative_sampling


def train(model, loader, optimizer, scheduler):
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
        if i % 100 == 0:
            torch.save(
                model.state_dict(), "drive/MyDrive/ALE_GNN/model_GCN_16.05_twitch"
            )
        scheduler.step()
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
        accuracy = accuracy_score(edge_label, pred)
        accuracies.append(accuracy)
    return np.average(f1scores), np.average(accuracies)
