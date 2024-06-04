import numpy as np
import pandas as pd
import torch  
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch import Tensor
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.loader import LinkLoader,DataLoader,LinkNeighborLoader
from torch_geometric.sampler import BaseSampler
import torch_geometric.transforms as T

from utils.preprocess_data import graph_data


class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # cosine similarity
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim = -1)

    def decode_all(self, z): 
        prob_adj = z @ z.t() # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t() # get predicted edge_list 


def train(train_data, train_loader, device, optimizer, model):
   
    """
    Single epoch model training in batches.
    :return: total loss for the epoch
    """
    
    model.train()
    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch.size()[0]
        z = model.forward(batch.x.float(), batch.edge_index)
        neg_edge_index = negative_sampling(edge_index = batch.edge_index, num_nodes = batch.num_nodes, num_neg_samples = None, method = 'sparse')
        edge_label_index = torch.cat([batch.edge_index, neg_edge_index], dim = -1, )
        edge_label = torch.cat([torch.ones(batch.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim = 0).to(device)
        out = model.decode(z, edge_label_index).view(-1)
        #loss = F.binary_cross_entropy_with_logits(out[:batch_size], edge_label[:batch_size])
        loss = F.binary_cross_entropy_with_logits(out, edge_label)
        # standard torch mechanics here
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch_size
    return total_loss



@torch.no_grad()
def test(loader):
    """
    Evalutes the model on the test set.
    :param loader: the batch loader
    :return: a score
    """
    model.eval()
    scores = []
    threshold = torch.tensor([0.7]).to(device)
    for batch in tqdm(loader):
        batch.to(device)
        z = model.forward(batch.x.float(), batch.edge_index)
        out = model.decode(z, batch.edge_index).view(-1).sigmoid()
        pred = (out > threshold).float() * 1
        score = f1_score(np.ones(batch.edge_index.size(1)), pred.cpu().numpy())
        scores.append(score)
    return np.average(scores)

if __name__ == '__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pd.read  
    edges, node_features, meme_dict, paper_ids_dict = graph_data(edges, node_features)
    model, train_data = Model(in_channels= np.shape(train_data.x)[1], hidden_channels=64).to(device), train_data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    for epoch in range(1, 10):
        loss = train()
        f1 = test(test_loader)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, f1: {f1:.5f}")
        


