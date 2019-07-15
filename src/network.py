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
        

