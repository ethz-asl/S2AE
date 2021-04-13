import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class L2Loss(nn.Module):
    """
    Simple L2 loss
    Takes the decoded data and a preprocessed segmentation
    """
    def __init__(self, margin, alpha, margin2):
        super(L2Loss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.margin2 = margin2

    def forward(self, decoded, teacher, size_average=True, batch_all=True):
        loss = (decoded - teacher).pow(2).sum(1)
        return loss
    

if __name__ == "__main__":
    print("Not yet implemented")
