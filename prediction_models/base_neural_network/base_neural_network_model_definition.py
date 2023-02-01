from torch import nn


class BaseNeuralNetworkModelDefinition(nn.Module):
    # TODO: maybe add as second base class to pred model
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(30*2, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 30*30),
            nn.InstanceNorm1d(30*30)
        )
        self.unflatten = nn.Unflatten(1, (30, 30))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        y = self.unflatten(logits)
        return y