import numpy as np
import torch
import time
#Change: the prediction is between node with change value and the rest of the dataset!!!
def accumulated_local_effects_exact(model, dataset, feature_index, num_bins=10, max_bin_size = None, k=256):
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
          data = dataset.clone()
          for idx in bin_data_idx_subset:

            model.eval()

            # Step 3: Calculate model predictions for lower and upper end of the section
            edge_label_index = torch.cat((torch.Tensor(np.random.choice(list(nodes), size=k)).int().unsqueeze(-1), idx*torch.ones(k).int().unsqueeze(-1)),dim=1)
            # LOOK HOW TO OPERATE ON TWO DATASET, CHANGING THE VALUE OF ONLY ONE
            old_value = data.x[idx, feature_index]
            data.x[idx, feature_index] = torch.tensor(bin_edges[bin_idx]).float()

            lower_encode = model(data.x, data.edge_index)
            lower = model.decode(lower_encode, edge_label_index).sigmoid()

            data.x[idx, feature_index] = torch.tensor(bin_edges[bin_idx + 1]).float()

            upper_encode = model(data.x, data.edge_index)
            upper = model.decode(upper_encode, edge_label_index).sigmoid()
            # Step 4: Subtract the above values. Average across all data points in the bin

            bin_ale.append(float(torch.mean(upper-lower).detach()))
            data.x[idx, feature_index] = old_value

          # Step 5: Calculate the cumulative sum of the averaged differences
          bin_ale = np.mean(bin_ale)
          if ale:
              ale.append(bin_ale + ale[-1])
          else:
              ale.append(bin_ale)

      end = time.time()
      return ale, end-start

#ale, t = accumulated_local_effects_exact(model,train_data, 0, 5, 64, k=16)


#Change: the prediction is between node with change value and the rest of the dataset!!!
def accumulated_local_effects(model, dataset, feature_index, num_bins=10, max_bin_size = None, k=256):
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
          data = dataset.clone()

          model.eval()

          # Step 3: Calculate model predictions for lower and upper end of the section
          unique = torch.Tensor(np.random.choice(list(nodes), size=k)).int()

          #create a tensor with edges to predict: for every node from bin_data_idx_subset check connections to every node in unique

          edge_label_index = torch.cat(
              (torch.Tensor(bin_data_idx_subset).int().repeat_interleave(unique.shape[0]).unsqueeze(-1),
               unique.repeat(bin_data_idx_subset.shape).int().unsqueeze(-1)), axis=1)
          old_values = data.x

          data.x[bin_data_idx_subset, feature_index] = torch.tensor(bin_edges[bin_idx]).float()
          lower_encode = model(data.x, data.edge_index)
          lower = model.decode(lower_encode, edge_label_index).sigmoid()

          data.x[bin_data_idx_subset, feature_index] = torch.tensor(bin_edges[bin_idx + 1]).float()
          upper_encode = model(data.x, data.edge_index)
          upper = model.decode(upper_encode, edge_label_index).sigmoid()
          # Step 4: Subtract the above values. Average across all data points in the bin

          bin_ale = float(torch.mean(upper-lower).detach())
          data.x = old_values

          if ale:
              ale.append(bin_ale + ale[-1])
          else:
              ale.append(bin_ale)

      end = time.time()
      return ale, end-start

