import torch
import pandas as pd
import numpy as np
import os
import logging
import torch_geometric
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.models import GraphSAGE, GCN, GAT
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.metrics import (
    LinkPredF1,
    LinkPredMAP,
    LinkPredPrecision,
    LinkPredRecall,
)

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.utils import negative_sampling
import sys
import psutil
from torch.utils.tensorboard import SummaryWriter
#from clearml import Task
#task = Task.init(project_name="Citation training", task_name="GCN")
if "/home/pkaczynska/repositories" not in sys.path:
    sys.path.append("/home/pkaczynska/repositories")

path_to_add = "C:/Users/ppaul/Documents"

if path_to_add not in sys.path:
    # Add the path to sys.path
    sys.path.append(path_to_add)
from influence_on_ideas.utils.preprocess_data import graph_data

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter('models/citations/tensorboard_logs2')

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
CONFIGS = [
    {
        "hidden_channels": 256,
        "lr": 0.0000005,#was 1e5?
        #'weight_decay': 5e-4,
        "epochs": 15,  # 4 left?
        "batch_size": 1024,
        "n_layers": 2,
        "model_type": "GAT",
    },
    {
        "hidden_channels": 256,
        "lr": 0.000001,
        #'weight_decay': 5e-4,
        "epochs": 15,
        "batch_size": 1024,
        "n_layers": 2,
        "model_type": "GCN",
    },
]  # GCN - 10 epok, GAT - 6

CONFIG = CONFIGS[0]


class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, model_type, n_layers):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.norm = BatchNorm(hidden_channels)
        if model_type == "GraphSAGE":
            self.model = GraphSAGE(
                in_channels,
                hidden_channels,
                n_layers,
                act=self.activation,
                norm=self.norm,
            )
        elif model_type == "GCN":
            self.model = GCN(
                in_channels,
                hidden_channels,
                n_layers,
                act=self.activation,
                norm=self.norm,
            )
        elif model_type == "GAT":
            self.model = GAT(
                in_channels,
                hidden_channels,
                n_layers,
                act=self.activation,
                norm=self.norm,
            )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.model(x, edge_index)

        return x

    def decode_batch(self, z, edge_label_index):
        preds = []
        batch_size = 1024
        for i in range(0, edge_label_index.size(1), batch_size):
            batch_edge_index = edge_label_index[:, i : i + batch_size]
            batch_pred = (z[batch_edge_index[0]] * z[batch_edge_index[1]]).sum(dim=-1)
            preds.append(batch_pred)
        return torch.cat(preds, dim=0)

    def decode(self, z, edge_label_index):

        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()  # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t()  # get predicted edge_list


def train(train_loader, device, optimizer, model, writer):
    """
    Single epoch model training in batches.
    :return: total loss for the epoch
    """
    model.train()
    total_examples = total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
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
        )
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
        # Log loss to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
        logging.debug(f"Loss: {loss.item()}")
        # scheduler.step()

    torch.save(
        model.state_dict(),
        os.path.join("models", "citations",
        CONFIG["model_type"]
        + ",n_layers"
        + str(CONFIG["n_layers"])
        + ",hidden_size"
        + str(CONFIG["hidden_channels"])),
    )
    print("Loss: ", total_loss)
    # return total_loss, total_examples


@torch.no_grad()
def test_0(model, loader, device="cuda"):
    """
    Evaluates the model on the test set.
    :param model: the model to evaluate
    :param loader: the batch loader
    :param device: the device to run the evaluation on (default: 'cuda')
    :return: a dictionary containing average F1 score and ROC AUC score
    """
    model.eval()

    f1 = LinkPredF1(1).to(device)
    recall = LinkPredRecall(1).to(device)
    precision = LinkPredPrecision(1).to(device)
    MAP = LinkPredMAP(1).to(device)
    threshold = torch.tensor([0.7]).to(device)

    for batch in tqdm(loader):
        batch.to(device)
        z = model(batch.x, batch.edge_index)  # Assuming model has __call__ method

        out = model.decode(z, batch.edge_label_index).view(-1).sigmoid()

        pred = (out > threshold).float().unsqueeze(1)

        labels = batch.edge_label.unsqueeze(1).type(torch.int64).to(device)
        f1.update(pred, labels)
        precision.update(pred, labels)
        recall.update(pred, labels)
        MAP.update(pred, labels)
        # all_preds.append(pred)
        # all_labels.append(batch.edge_label)
    avg_f1 = f1.compute()
    avg_precision = precision.compute()
    avg_recall = recall.compute()
    avg_MAP = MAP.compute()
    # all_preds = torch.concat(all_preds).detach().cpu().numpy()
    # all_labels = torch.concat(all_labels).detach().cpu().numpy()

    # avg_f1 = f1_score(all_labels, all_preds)
    # avg_roc_auc = roc_auc_score(all_labels, all_preds)

    logging.info(
        f"F1 score: {avg_f1:.4f}, MAP: {avg_MAP:.4f}, recall: {avg_recall:.4f}, precission: {avg_precision:.4f}"
    )
    print(
        f"F1 score: {avg_f1:.4f}, MAP: {avg_MAP:.4f}, recall: {avg_recall:.4f}, precission: {avg_precision:.4f}"
    )
    return {
        "F1 score": avg_f1,
        "MAP": avg_MAP,
        "recall": avg_recall,
        "precission": avg_precision,
    }


@torch.no_grad()
def test(model, loader, device="cuda"):
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

    for batch in loader:
        batch.to(device)
        z = model(batch.x, batch.edge_index)  # Assuming model has __call__ method

        out = model.decode(z, batch.edge_label_index).view(-1).sigmoid()

        #pred = (out > threshold).float()

        all_preds.append(out)
        all_labels.append(batch.edge_label)

    all_preds = torch.concat(all_preds)
    all_labels = torch.concat(all_labels)
    
    avg_roc_auc = roc_auc_score(all_labels, all_preds)
    
    avg_f1 = f1_score(all_labels, (all_preds > threshold).float())
    
    # Log metrics to TensorBoard


    #logging.info(f"F1 score: {avg_f1:.4f}, ROC AUC: {avg_roc_auc:.4f}")
    print(f"F1 score: {avg_f1:.4f}, ROC AUC: {avg_roc_auc:.4f}")
    return {"average_f1_score": avg_f1, "average_roc_auc": avg_roc_auc}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on: ", device)
    edges_path = "data/citations/edge.parquet"
    node_features_path = "data/citations/node_features.parquet"
    data_path = "data/citations/"
    model_path = f"models/citations/{CONFIG['model_type']},n_layers{CONFIG['n_layers']},hidden_size{CONFIG['hidden_channels']}"

    logger = logging.getLogger(__name__)
    # logging.basicConfig(filename=model_path+'.log', encoding='utf-8', level=logging.DEBUG)
    edges = pd.read_parquet(edges_path)
    node_features = pd.read_parquet(node_features_path)
    train_loader, test_loader, train_data, test_data, val_data, val_loader = graph_data(
        edges, node_features, "data/citations/"
    )
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=model_path+'.log', encoding='utf-8', level=logging.DEBUG)
    # Initialize the model
    model = Model(
        in_channels=np.shape(train_data.x)[1],
        hidden_channels=CONFIG["hidden_channels"],
        model_type=CONFIG["model_type"],
        n_layers=CONFIG["n_layers"],
    ).to(device)

    # Check if the model weights file exists
    if os.path.isfile(model_path):
        # Load the weights into the model
        model.load_state_dict(torch.load(model_path))
        print("Model weights loaded successfully.")
    else:
        print(
            "Model weights file does not exist. Initializing model with random weights."
        )
    #params = task.connect_configuration(CONFIG)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=CONFIG["lr"])
    scheduler = None  # ExponentialLR(optimizer, gamma=0.99)
    loss_values = []
    print(model_path)
    valmetrics = test(model, val_loader, device)
    testmetrics = test(model, test_loader, device)
    print(valmetrics, testmetrics)
    #writer.add_scalar('Test/average_f1_score', metrics['average_f1_score'], 0)
    #writer.add_scalar('Test/average_roc_auc', metrics['average_roc_auc'], 0)
    #logging.info(metrics)
    #for epoch in range(CONFIG["epochs"]):
    #    print(f"Epoch: {epoch}")
    #    train(train_loader, device, optimizer, model, writer)
    #    metrics = test(model, test_loader, device)
        #writer.add_scalar('Test/average_f1_score', metrics['average_f1_score'], epoch+1)
        #writer.add_scalar('Test/average_roc_auc', metrics['average_roc_auc'], epoch+1)
    #    logging.info(metrics)
# Close the TensorBoard SummaryWriter after training
#writer.close()