import random
import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm

from influence_on_ideas.utils.preprocess_data import graph_data
from influence_on_ideas.models.gnn import Model

def graph_pdp_exact(dataset, model, column, values, max_bin_size=256, k=256, device):
    '''function to plot the partial dependence plot
    data: ??
    model: the pyg model for link prediction
    column: the column index
    values: list of values to check the partial dependence for
    Example: How much probable is on average to cite a paper if it is coauthored solely by Big tech authors than having a mixed affiliation?
    '''
    start = time.time()
    probabilities = {}
    random_papers = np.random.choice(range(dataset.x.shape[0]), size=max_bin_size, replace=False)

    model.eval()
    with torch.no_grad():
      for value in tqdm(values):
        probabilities[value] = []
        for paper in random_papers:
            dataset_pdp = dataset.clone()
            dataset_pdp.x[paper, column] = torch.tensor(value)
            unique = np.random.choice(range(dataset.x.shape[0]), size=k, replace=False)
            edge_label_index = torch.cat((torch.Tensor(unique).unsqueeze(-1), int(paper)*torch.ones(unique.shape).unsqueeze(-1)),dim=1).type(torch.LongTensor).to(device)

            z = model.forward(dataset_pdp.x, dataset_pdp.edge_index)
            out = model.decode(z, edge_label_index).view(-1).sigmoid()

            probabilities[value].append(out.detach().cpu().numpy())
        probabilities[value] = np.mean(probabilities[value])
    end = time.time()
    #plt.plot(values, probabilities.values())
    #plt.xlabel(column)
    #plt.ylabel('Mean probability of being cited')
    return probabilities, end-start

def graph_pdp_approximate(dataset, model, column, values, max_bin_size=256, k=256, device):
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

      dataset.x[random_papers, column] = torch.tensor(value).float()
      unique = np.random.choice(range(dataset.x.shape[0]), size=k, replace=False)

      edge_label_index = torch.cat((torch.Tensor(unique).int().repeat(random_papers.shape[0]).unsqueeze(-1), torch.Tensor(random_papers).int().repeat_interleave(unique.shape[0]).unsqueeze(-1)),dim=1).to(device)

      model.eval()
      with torch.no_grad():

        z = model.forward(dataset.x, dataset.edge_index)
        out = model.decode(z, edge_label_index).view(-1).sigmoid()

        probabilities[value].append(np.mean(out.detach().cpu().numpy()))
    end = time.time()

    return probabilities, end-start

if __name__ == '__main__':
  # read data. read model
  for _ in range(20):
    ale_exact, t_exact = graph_pdp_exact(model,train_data, 0, 5, None)
    ale_approximate, t_approximate = graph_pdp_approximate(model,train_data, 0, 5, None)
    results = pd.concat([results, pd.DataFrame({'k': 256, 'max_bin_size': 'max', 'PDP exact': ale_exact, 'time_exact': t_exact, 'PDP approximate': ale_approximate, 'time_approximate': t_approximate})])
    # check if it saves if no file is present
    results.to_csv('data/PDP_comparison.csv', mode='a', header=False)#, 'time_approximate': t_approximate'ALE approximate': ale_approximate,