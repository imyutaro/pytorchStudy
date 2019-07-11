import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):
    def __init__(self, nItem, mid_dim):
        super(TripletNet, self).__init__()
        self.embed = nn.Linear(nItem, mid_dim)
        self.fc = nn.Linear(mid_dim, 2)

    def embedding(self, x):
        x = F.relu(self.embed(x))
        x = self.fc(x)
        return x
    
    def forward(self, input1, input2):
        output1 = self.embedding(input1)
        output2 = self.embedding(input2)
        return output1, output2
        

def siameseLoss(output, target):
    """
    L(d, t) := t*d^2 + (1-t)*max(0, margin-d)^2
    d := ||y-y'||_2, t := \delta(c, c') \in {0, 1}
    y is a embedding vector from x and y' is a embedding vector from x'.
    """
    pass


def tripletLoss():
    pass
