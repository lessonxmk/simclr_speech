import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import resnet


class Encoder(nn.Module):
    def __init__(self, model='resnet18'):
        super(Encoder, self).__init__()
        if (model == 'resnet18'):
            self.model = resnet.ResNet(resnet.ResidualBlock)
        if(model=='resnet50'):
            self.model=models.resnet50()

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
            self.mlp1 = nn.Linear(input_shape[-1], out_size)

    def forward(self, *input):
        x = input[0]
        if (self.mode == 'nonlinear'):
            x = self.mlp0(x)
            x = F.relu(x)
            x = self.mlp1(x)
        return x


class contrastiveLoss(nn.Module):
    def __init__(self,
                 hidden_norm=True,
                 temperature=1.0,
                 weights=1.0,
                 LARGE_NUM=1e9):
        super().__init__()
        self.hidden_norm = hidden_norm
        self.temperature = temperature
        self.weights = weights
        self.LARGE_NUM = LARGE_NUM

    def forward(self, hidden):
        if self.hidden_norm:
            hidden = F.normalize(hidden, 2, -1)
        hidden1, hidden2 = torch.split(hidden, hidden.shape[0] // 2, 0)
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1.transpose(1, 0)
        hidden2_large = hidden2.transpose(1, 0)
        labels = torch.arange(0, batch_size)
        masks = F.one_hot(torch.arange(0, batch_size), batch_size).float()
        if torch.cuda.is_available():
            labels = labels.cuda()
            masks = masks.cuda()

        logits_aa = torch.matmul(hidden1, hidden1_large) / self.temperature
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large) / self.temperature
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large) / self.temperature
        logits_ba = torch.matmul(hidden2, hidden1_large) / self.temperature

        logits_a = torch.cat((logits_ab, logits_aa), 1)
        logits_b = torch.cat((logits_ba, logits_bb), 1)

        loss_a = F.cross_entropy(logits_a, labels) * self.weights
        loss_b = F.cross_entropy(logits_b, labels) * self.weights
        loss = loss_a + loss_b
        return loss
