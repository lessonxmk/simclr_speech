import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
from absl import flags

import dataloader
import augment
import model

FLAGS = flags.FLAGS


def loadDataPath(format='wav', *path):
    dataPath = []
    for p in path:
        dataPath.append(glob.glob(p) + '.' + format)
    return dataPath()


def Augment(x, *operations):
    availableOperation = {
        'crop': augment.crop,
        'gwn': augment.Gaussian_white_noise,
    }
    for op in operations:
        x = availableOperation[op](x)
    return x


def run():
    train_data = loadDataPath()
    train_data = dataloader.DataSet(train_data)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    encoder = model.Encoder()
    # projectionHead = model.projectionHead()
    projectionHead = None
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        # projectionHead = projectionHead.cuda()
    criterion = model.contrastiveLoss()
    optimizerE = optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizerP = optim.Adam(projectionHead.parameters(), lr=learning_rate)
    for epoch in range(Epochs):
        print_loss = 0
        for _, x in enumerate(train_loader):
            if torch.cuda.is_available():
                x = x.cuda()

            x1 = Augment(x)
            x2 = Augment(x)
            x = torch.cat((x1, x2), 0)

            representation = encoder(x)
            if (projectionHead is None):
                projectionHead = model.projectionHead(x.shape)
                if torch.cuda.is_available():
                    projectionHead = projectionHead.cuda()
            out = projectionHead(representation)

            loss = criterion(out)
            print_loss += loss.data.item() * BATCH_SIZE
            optimizerE.zero_grad()
            optimizerP.zero_grad()
            loss.backward()
            optimizerE.step()
            optimizerP.step()


if __name__ == '__main__':
    run()
