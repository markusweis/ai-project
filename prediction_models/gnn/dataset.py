from typing import Dict, List, Set, Tuple

import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset

from graph import Graph
from torch_geometric.data import Data




class CustomGraphDataset(Dataset):
    """returns a singe Graph for the GNNStack to read as a tuple of 
        node features(id, familyId) and edge_indexs 
        Problem: Because edge_index does not have the same dimension, the batchsize
        has to be one, because torch can't stack the tensors. 
        Maybe return the adjecency matrix instead and make the conversion later. 

    """
    def __init__(self, graphs: List[Graph]):
        self.graphs = graphs
        self.parts_lists = [list(graph.get_parts()) for graph in self.graphs]
        # Sort the parts to reduce possible combinations
        for parts_list in self.parts_lists:
            parts_list.sort()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        parts_list_tensor = [(part.get_part_id(), part.get_family_id()) for part in self.parts_lists[idx]]
        adj_matr_tensor = torch.tensor(self.graphs[idx].get_adjacency_matrix(part_order=self.parts_lists[idx]), dtype=torch.float32)
        edge_index = adj_matr_tensor.nonzero().t().contiguous()
        
        # if torch.geometric functions are every necessary, it is possible to convert like this: 
        # data = Data(x=parts_list_tensor, edge_index=edge_index)

    
        return parts_list_tensor, edge_index
