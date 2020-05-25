import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, model='resnet18'):
        super(Encoder, self).__init__()
        if (model == 'resnet18'):
            self.model = models.resnet18()

    def forward(self, *input):
        x = input[0]
        x = self.model(x)
        return x


class projectionHead(nn.Module):
    def __init__(self, input_shape, out_size=128, mode='nonlinear'):
        super(projectionHead, self).__init__()
        self.mode = mode
        if (self.mode == 'nonlinear'):
            self.mlp0 = nn.Linear(input_shape[-1], input_shape[-1])
            self.nlp1 = nn.Linear(input_shape[-1], out_size)

    def forward(self, *input):
        x = input[0]
        if (self.mode == 'nonlinear'):
            x = self.mlp0(x)
            x = self.mlp1(x)
        return x


class contrastiveLoss(nn.Module):
    def __init__(self,
                 hidden_norm=True,
                 temperature=1.0,
                 weights=1.0,
                 LARGE_NUM=1e9):
        super().__init__()
        self.hidden_norm = hidden_norm,
        self.temperature = temperature,
        self.weights = weights
        self.LARGE_NUM=LARGE_NUM
    def forward(self, hidden):
        

        return
