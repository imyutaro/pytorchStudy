import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss defined: 
    L(d, t) := t*d^2 + (1-t)*max(0, margin-d)^2
    d := ||y-y'||_2, 
    t := \delta(c, c') \in {0, 1}

    y is a embedding vector from x and y' is a embedding vector from x'.
    If x and x' are in same class, t=1 else t=0.
    """
    def __init__(self, margin):
        super(ConstrastiveLoss, self).__init__()
        self.margin = margin

    def __call__(output1, output2, target1, target2, t=0):
        dis = torch.norm(output1-output2)
        loss = 0.5*t*dis + 0.5*(1-t)*F.relu(self.margin-dis).pow(2)
