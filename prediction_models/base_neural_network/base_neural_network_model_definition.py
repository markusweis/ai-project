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

        self.linear_relu_stack.append(nn.Linear(meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH*2, meta_parameters.HIDDEN_LAYERS_SIZE))
        self.linear_relu_stack.append(nn.ReLU())

        for i in range(meta_parameters.NUM_HIDDEN_LAYERS):
            self.linear_relu_stack.append(nn.Linear(meta_parameters.HIDDEN_LAYERS_SIZE, meta_parameters.HIDDEN_LAYERS_SIZE))
            self.linear_relu_stack.append(nn.ReLU())

        self.linear_relu_stack.append( nn.Linear(meta_parameters.HIDDEN_LAYERS_SIZE, meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH**2),)
        self.linear_relu_stack.append(nn.InstanceNorm1d(meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH**2))

       
        self.unflatten = nn.Unflatten(1, (meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH, meta_parameters.MAX_NUMBER_OF_PARTS_PER_GRAPH))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        y = self.unflatten(logits)
        return y