import os 
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from torch_geometric.loader import LinkNeighborLoader

from sklearn.preprocessing import normalize, MultiLabelBinarizer

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch import Tensor

import warnings


from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch import Tensor
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def preprocess_data_citations(df):
    # Generate dataframes of the edges and node features, and the meme dictionary
    df.reset_index(inplace=True)
    id_counts = (
        df["noun_chunks_cleaned"].explode().value_counts()
    )  # or meme_ids or noun_chunks_cleaned
    ids_appearing_more = id_counts[id_counts > 800].index.tolist() # frequency threshold - can be adjusted

    print("Number of memes appearing more than 800 times:", len(ids_appearing_more))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlb = MultiLabelBinarizer(classes=ids_appearing_more)  # , sparse_output=True)
        meme_one_hot = mlb.fit_transform(df["noun_chunks_cleaned"])  # memes_ids/noun_chunks_cleaned

    # dict from the position to name in the position
    meme_dict = {i: meme for i, meme in enumerate(list(mlb.classes_))}
    print("further preprocessing")
    edges = df.explode(column="outbound_citations")[
        ["paper_id", "outbound_citations"]
    ].dropna()
    edges["outbound_citations"] = edges["outbound_citations"].apply(int)
    # edges = df_edge.merge(df['paper_id'],left_on='outbound_citations',right_on='paper_id',how='inner').drop(columns=['paper_id_y']).rename(columns={'paper_id_x':'paper_id'})
    paper_ids_dict = {paper_id: index for index, paper_id in enumerate(df["paper_id"])}
    edges["paper_id"] = edges["paper_id"].apply(lambda x: paper_ids_dict[x])
    edges["outbound_citations"] = edges["outbound_citations"].apply(
        lambda x: paper_ids_dict[x] if x in paper_ids_dict.keys() else None
    )
    edges = edges.dropna(subset=["outbound_citations"])
    print('columns', df.columns)
    node_features = pd.concat(
        (
            df[["BT_percent", "company", "authors_number", "double_affiliation"]],
            pd.DataFrame(meme_one_hot, columns=mlb.classes_),
        ),
        axis=1,
    )

    return edges, node_features, meme_dict, paper_ids_dict

def graph_data(edges, node_features, path, batch_size=1024):

    train_data_path = path+'train_data.pt'
    test_data_path = path+'test_data.pt'
    val_data_path = path+'val_data.pt'
    edge_array = edges.values.T
    edge_index = torch.tensor(edge_array, dtype=torch.int64)

    if os.path.isfile(train_data_path) and os.path.isfile(test_data_path) and os.path.isfile(val_data_path):
        # Load the train/test data
        train_data = torch.load(train_data_path)
        test_data = torch.load(test_data_path)
        val_data = torch.load(val_data_path)
        print("Train and test data loaded successfully.")
    else:
        data = Data(x=torch.tensor(node_features.values.astype(np.float32)),
                    edge_index=edge_index,
                    #y=node_features.company,
                    )

        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            #disjoint_train_ratio=0.3,
            #neg_sampling_ratio=2.0,
            add_negative_train_samples=False
        )
        
        train_data, val_data, test_data = transform(data)
        torch.save(train_data, train_data_path)
        torch.save(test_data, test_data_path)
        torch.save(val_data, val_data_path)
    train_data, val_data, test_data = train_data, val_data, test_data

    # Define seed edges:
    edge_label_index = train_data.edge_label_index
    edge_label = train_data.edge_label


    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=[30] * 2,
        batch_size=batch_size,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
    )
    test_loader = LinkNeighborLoader(
        test_data,
        num_neighbors=[30] * 2,
        batch_size=batch_size,
        edge_label_index=test_data.edge_label_index,
        edge_label=test_data.edge_label,
    )
    return train_loader, test_loader, train_data, test_data  

if __name__=='__main__':
    df = pd.read_parquet(r"C:\Users\ppaul\Documents\AI-strategies-papers-regulations-monitoring\data\s2orc\big_ai_dataset_with_affiliations_extended_oa.parquet")
    edges, node_features, meme_dict, paper_ids_dict = preprocess_data_citations(df)
    edges.to_csv("./data/citations/edge.csv")
    pd.DataFrame(node_features).to_csv("./data/citations/node_features.csv")