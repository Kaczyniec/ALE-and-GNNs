import random
import torch
import numpy as np
def graph_pdp_cat(dataset, model, column, values):
    '''function to plot the partial dependence plot
    data: ??
    model: the pyg model for link prediction
    column: the column index
    values: list of values to check the partial dependence for
    Example: How much probable is on average to cite a paper if it is coauthored solely by Big tech authors than having a mixed affiliation?
    '''
    #TODO: change the values separately and compute the probability of citation for only the changed values!
    #TODO: check if the first is cited or citing?
    probabilities = {}
    random_papers = random.choices(range(dataset.__len__()), k=100)
    
    for paper in random_papers:
        
        for value in values:
            
            dataset.x[paper, column] = torch.tensor(value)
            unique = dataset.edge_index.unique().unsqueeze(1)[:256,:]
            #paper should not be in the unique!!!
            edge_label_index = torch.cat((unique, int(paper)*torch.ones(unique.shape)),dim=1).type(torch.LongTensor).to(device)
            dataset = dataset.to(device)

            val_loader = LinkNeighborLoader(
                        dataset,
                        num_neighbors=[30] * 2,
                        batch_size=256,
                        edge_label_index=edge_label_index,
                        edge_label=dataset.edge_label,
                    )
            
            probabilities[value] = []

            for batch in val_loader:
                z = model.forward(batch.x.float(), batch.edge_index)
                out = model.decode(z, batch.edge_label_index).view(-1).sigmoid()
                
                probabilities[value] = np.concatenate((probabilities[value],out.detach().cpu().numpy()),axis=0)
            probabilities[value] = probabilities[value].mean()
    print(probabilities)
    plt.plot(values, probabilities.values())
    plt.xlabel(column)
    plt.ylabel('Mean probability of being cited')