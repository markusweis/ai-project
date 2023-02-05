import torch
from typing import List, Tuple
from torch.utils.data import Dataset
from torch.nn.functional import pad

from graph import Graph
from part import Part
from prediction_models.base_neural_network import meta_parameters

class GraphDataset(Dataset):
    def __init__(self, graphs: List[Graph]):
        self.graphs:List[Graph] = graphs
        self.parts_lists = [list(graph.get_parts()) for graph in self.graphs]
        # Sort the parts to reduce possible combinations
        for parts_list in self.parts_lists:
            parts_list.sort()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
    
        parts_tensor = self.get_parts_tensor(parts_list=self.parts_lists[idx])

        nonredundand_connections_array_tensor = torch.tensor(self.graphs[idx].get_nonredundand_connections_array(
            part_order=self.parts_lists[idx], 
            pad_to_node_count = meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH,  
            padding_value=meta_parameters.UNUSED_NODES_PADDING_VALUE))

        return parts_tensor, nonredundand_connections_array_tensor

    @classmethod
    def get_parts_tensor(cls, parts_list: List[Part], extra_dimension:bool = False) -> torch.Tensor:
        """
        Creates the input tensor for the model
        :param extra_dimension: Whether to embed the  tensor in an extra dimension simulating the same tensor 
        shape as for a batch input
        """
        # Range of ids (+ 2 because id 0 can exist and last is reserved for node padding)
        num_part_ids = meta_parameters.MAX_SUPPORTED_PART_ID + 2
        padding_part_id = meta_parameters.MAX_SUPPORTED_PART_ID + 1
        num_family_ids = meta_parameters.MAX_SUPPORTED_FAMILY_ID + 2
        padding_family_id = meta_parameters.MAX_SUPPORTED_FAMILY_ID + 1

        # Feature values
        part_ids = [part.get_part_id() for part in parts_list]
        family_ids = [part.get_family_id() for part in parts_list]
        
        # Padding to achieve the same size for each input
        missing_node_count = meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH - len(parts_list)
        padding_list_part_ids = [padding_part_id for _ in range(missing_node_count)]
        padding_list_fam_ids = [padding_family_id for _ in range(missing_node_count)]
        part_ids.extend(padding_list_part_ids)
        family_ids.extend(padding_list_fam_ids)
        
        # Feature tensors
        part_ids_tensor = torch.tensor(part_ids)
        family_ids_tensor = torch.tensor(family_ids)

        # On hot encodings
        part_ids_one_hot = torch.nn.functional.one_hot(part_ids_tensor, num_classes=num_part_ids)
        family_ids_one_hot = torch.nn.functional.one_hot(family_ids_tensor, num_classes=num_family_ids)
        
        # Combine both tensors
        combined_tensor = torch.cat((part_ids_one_hot, family_ids_one_hot), 1)

        # Change the type to match other layers
        combined_tensor = combined_tensor.type(torch.float32)

        # Add the extra dimension, if requested
        if extra_dimension:
            return combined_tensor[None, :]
        else:
            return combined_tensor