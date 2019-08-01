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
    If x and x' are in same class(c==c'), t=1 else t=0.
    """
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def __call__(self, y1, y2, t):
        dis = torch.norm(y1-y2)
        loss = 0.5*t*dis.pow(2) + 0.5*(1-t)*F.relu(self.margin-dis).pow(2)
        return loss

class TripletLoss(nn.Module):
    """
    """
    def __init__(self):
        super(TripletLoss, self).__init__()

    def __call__(self, y, pos, neg):
        """
        Triplet loss is defined:
        L(y, y^+, y^-) := max{0, g+D(y, y^+)-D(y, y^-)}

        D(f(a), f(b)) = ||a-b||^2_2
        f(.) is the embedding function.
        y is a embedding vector from x (y = f(x)).
        x^+ is similar sample to x (positive sample).
        x^- is not similar sample to x (negative sample).
        g is a gap regularization parameter same as margin(distance between pos and neg).

        I don't know what g means...

        minimize loss:
        min L(y, y^+, y^-)+\lambda*||W||^2_2

        W is the parameters of the embedding function f(.).
        In the paper \lambda = 0.001.
        \lambda is a regularization parameter.

        ref: https://arxiv.org/pdf/1404.4661.pdf
        """

        pos_dis = torch.norm(y, pos).pow(2)
        neg_dis = torch.norm(y, neg).pow(2)

        loss = F.relu(pos_dis - neg_dis)
        return loss

    def faceloss(self):
        pass

