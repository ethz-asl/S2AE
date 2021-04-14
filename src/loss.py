import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class L2Loss(nn.Module):
    """
    Simple L2 loss
    Takes the decoded data and a preprocessed segmentation
    """
    def __init__(self, alpha, margin):
        super(L2Loss, self).__init__()
        self.alpha = alpha
        self.margin = margin

    def forward(self, decoded, teacher, size_average=True, batch_all=True):
        distance_dec_teacher = (decoded - teacher).pow(2).sum(1)
        losses = F.relu(distance_dec_teacher - self.margin) + self.alpha

        if size_average:
            if batch_all:
                return distance_dec_teacher, losses.sum()/(((losses > 1e-16).sum()).float()+1e-16), losses.mean()
            else:
                return distance_dec_teacher, losses.mean()
        else:
            return distance_positive, losses.sum()


if __name__ == "__main__":
    criterion = L2Loss(alpha=0.5, margin=0.2)
