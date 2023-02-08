from typing import Dict, List, Set, Tuple

import torch
from torch.nn.functional import pad, one_hot
from torch.utils.data import Dataset

from graph import Graph
from torch_geometric.data import Data

from prediction_models.gnn.meta_parameters import MAX_SUPPORTED_PART_ID




class CustomGraphDataset(Dataset):
    """returns a singe Graph for the GNNStack to read as a tuple of 
        node features(id, familyId) and edge_indexs 
        Problem: Because edge_index does not have the same dimension, the batchsize
        has to be one, because torch can't stack the tensors. 
        Maybe return the adjecency matrix instead and make the conversion later. 

    """
    def __init__(self, graphs: List[Graph]):
        print("Preparing dataset...")
        self.graphs = graphs
        self.parts_lists = [list(graph.get_parts()) for graph in self.graphs]

        self.dataset = []
        for p, g in zip(self.parts_lists, self.graphs):
            parts_tensor = torch.tensor([(part.get_part_id(), part.get_family_id()) for part in p])

            adj_matr_tensor = torch.tensor(g.get_adjacency_matrix(part_order=p), dtype=torch.float32)    
            self.dataset.append((parts_tensor, adj_matr_tensor))
        print("Finished preparing dataset.")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx): 
        parts_tensor, adj_matr = self.dataset[idx]  
        one_hot_encoded = one_hot(parts_tensor, MAX_SUPPORTED_PART_ID + 1)
        one_hot_flattened = torch.flatten(one_hot_encoded, start_dim=1).float()

        # create labels for each edge
        num_nodes = adj_matr.size(dim=0)
        indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
        labels = adj_matr[indices[0], indices[1]]
        return one_hot_flattened, labels


