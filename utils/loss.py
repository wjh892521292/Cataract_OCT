import torch
import torch.nn as nn
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class CVALoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x, y):
        loss = (1 - y) * self.relu(x) + y * self.relu(-x)
        return loss.mean(dim=0)


class CVALoss2(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x, y):
        loss = self.relu((y-x)*(y-x)-0.01)
        return loss.mean()