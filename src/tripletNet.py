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
    
    def forward(self, x1, x2):
        y1 = self.embedding(x1)
        y2 = self.embedding(x2)
        return y1, y2

    def get_embedding(x):
        return self.embedding(x)
        

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss defined: 
    L(d, t) := t*d^2 + (1-t)*max(0, margin-d)^2
    d := ||y-y'||_2, t := \delta(c, c') \in {0, 1}
    y is a embedding vector from x and y' is a embedding vector from x'.
    """
    def __init__(self, margin):
        super(ConstrastiveLoss, self).__init__()
        self.margin = margin

    def __call__(output1, output2, target1, target2, t=0):
        dis = torch.norm(output1-output2)
        loss = 0.5*t*dis + 0.5*(1-t)*F.relu(self.margin-dis).pow(2)
        return loss


def tripletLoss():
    pass
