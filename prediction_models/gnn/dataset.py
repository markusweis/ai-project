from typing import Dict, List, Set, Tuple

import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset

from graph import Graph
from torch_geometric.data import Data

from prediction_models.gnn.constants import MAX_NUMBER_OF_PARTS_PER_GRAPH




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
        parts_tensor = torch.tensor([(part.get_part_id(), part.get_family_id()) for part in self.parts_lists[idx]])
        adj_matr_tensor = torch.tensor(self.graphs[idx].get_adjacency_matrix(part_order=self.parts_lists[idx]), dtype=torch.float32)
        edge_index = adj_matr_tensor.nonzero().t().contiguous().long()
        
        
         # Padding to achieve the same size for each input
        missing_node_count = MAX_NUMBER_OF_PARTS_PER_GRAPH - len(self.parts_lists[idx])
        if missing_node_count > 0:
            parts_tensor = pad(parts_tensor, (0, 0, 0, missing_node_count), "constant", -1).long()
    
        
        return parts_tensor, edge_index
