from collections import OrderedDict
from torch import nn
from prediction_models.base_neural_network import meta_parameters


class BaseNeuralNetworkModelDefinition(nn.Module):
    def __init__(self):
        super().__init__()
        assert (meta_parameters.NUM_HIDDEN_LAYERS >= 1), 'Number of layers is not >=1'
        assert (meta_parameters.HIDDEN_LAYERS_SIZE >= 1), 'Size of hidden layers is not >=1'

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential()

        # Range of ids (+ 2 because id 0 can exist and last is reserved for node padding)
        num_part_ids = meta_parameters.MAX_SUPPORTED_PART_ID + 2
        num_family_ids = meta_parameters.MAX_SUPPORTED_FAMILY_ID + 2

        self.linear_relu_stack.append(nn.Linear(meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH*(num_part_ids+num_family_ids), meta_parameters.HIDDEN_LAYERS_SIZE))
        self.linear_relu_stack.append(nn.ReLU())

        for i in range(meta_parameters.NUM_HIDDEN_LAYERS):
            self.linear_relu_stack.append(nn.Linear(meta_parameters.HIDDEN_LAYERS_SIZE, meta_parameters.HIDDEN_LAYERS_SIZE))
            self.linear_relu_stack.append(nn.ReLU())
        output_size = self._compute_output_size(meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH)
        self.linear_relu_stack.append( nn.Linear(meta_parameters.HIDDEN_LAYERS_SIZE, output_size),)
       
    def forward(self, x):
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        return y

    def _compute_output_size(self, node_count: int) -> int:
        size = 0
        # Iterate through rows:
        for row in range(node_count - 1):
            size += node_count - (row + 1)

        return size