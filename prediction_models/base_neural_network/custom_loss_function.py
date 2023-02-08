
import torch
from torch import nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, targets):

        # standard loss
        # loss = F.mse_loss(input=inputs,target=targets)

        # Filter only actual nodes
        mask = torch.where(targets >= 0, True, False)

        masked_inputs = torch.masked_select(input=inputs, mask=mask)
        masked_targets = torch.masked_select(input=targets, mask=mask)

        # Loss only on the actual edges
        loss = F.mse_loss(input=masked_inputs,target=masked_targets)
                       
        return loss