import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):
    def __init__(self, nItem, mid_dim):
        super(Siamese, self).__init__()
        self.embed = nn.Linear(nItem, nItem//4)
        self.fc = nn.Linear(nItem//4, mid_dim)

    def embedding(self, x):
        x = F.relu(self.embed(x))
        x = self.fc(x)
        return x
    
    def forward(self, x1, x2):
        y1 = self.embedding(x1)
        y2 = self.embedding(x2)
        return y1, y2

    def get_embedding(self, x):
        return self.embedding(x)
        

class ConvSiamese(nn.Module):
    def __init__(self):
        super(ConvSiamese, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x1, x2):
        y1 = self.conv(x1)
        y1 = y1.view(y1.size(0), -1)
        y1 = self.fc(y1)
        y2 = self.conv(x2)
        y2 = y2.view(y2.size(0), -1)
        y2 = self.fc(y2)
        return y1, y2

    def get_embedding(self, x):
        y = self.conv(x)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y.squeeze()
