import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletNet(nn.Embedding):
    def __init__(self, nItem):
        super(TripletNet, self).__init__()
        self.nItem = nItem
        self.emb = nn.Embedding(nItem)
        self.fc = nn.Linear(, 1)

    def forward(self, x):
        pass


def siameseLoss(output, target):
    pass


def tripletLoss():
    pass
