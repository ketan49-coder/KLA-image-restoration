import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.l1(pred, target) + 0.1 * self.mse(pred, target)
