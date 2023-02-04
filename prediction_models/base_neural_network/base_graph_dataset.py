import torch
from typing import List, Tuple
from torch.utils.data import Dataset
from torch.nn.functional import pad

from graph import Graph
from part import Part
from prediction_models.base_neural_network import meta_parameters

class BaseGraphDataset(Dataset):
    def __init__(self, graphs: List[Graph]):
        self.graphs:List[Graph] = graphs
        self.parts_lists = [list(graph.get_parts()) for graph in self.graphs]
        # Sort the parts to reduce possible combinations
        for parts_list in self.parts_lists:
            parts_list.sort()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
    
        parts_tensor = torch.tensor([[part.get_part_id(), part.get_family_id()] for part in self.parts_lists[idx]], dtype=torch.float32)
        nonredundand_connections_array_tensor = torch.tensor(self.graphs[idx].get_nonredundand_connections_array(
            part_order=self.parts_lists[idx], 
            pad_to_node_count = meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH,  
            padding_value=meta_parameters.UNUSED_NODES_PADDING_VALUE))

        # Padding to achieve the same size for each input
        missing_node_count = meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH - len(self.parts_lists[idx])
        if missing_node_count > 0:
            parts_tensor = pad(parts_tensor, (0, 0, 0, missing_node_count), "constant", meta_parameters.UNUSED_NODES_PADDING_VALUE)

        return parts_tensor, nonredundand_connections_array_tensor