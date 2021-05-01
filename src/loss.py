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
                return losses.sum()/(((losses > 1e-16).sum()).float()+1e-16), losses.mean()
            else:
                return losses.mean()
        else:
            return losses.sum()

        
class CrossEntropyLoss(nn.Module):
    def __init__(self, n_classes):
        super(CrossEntropyLoss, self).__init__()
        
        # define a weighted loss (0 weight for 0 label)
        weights_list = [0]+[1 for i in range(n_classes)]
        weights = np.asarray(weights_list)
        self.weight_torch = torch.Tensor(weights_list)

    def forward(self, decoded, teacher, size_average=True, batch_all=True):
        losses = F.cross_entropy(decoded, teacher, weight=None, size_average=size_average, ignore_index=250)

        if size_average:
            if batch_all:
                return losses.sum()/(((losses > 1e-16).sum()).float()+1e-16), losses.mean()
            else:
                return losses.mean()
        else:
            return losses.sum()
        
class NegativeLogLikelihoodLoss(nn.Module):
    def __init__(self, n_classes):
        super(NegativeLogLikelihoodLoss, self).__init__()        
        self.n_classes = n_classes
        
    def forward(self, decoded, teacher, size_average=True, batch_all=True):
        print(f"shape of decoded is {decoded.shape} and teacher is {teacher.shape}")
        losses = F.nll_loss(decoded, teacher, weight=None, size_average=size_average, reduction='none')

        if size_average:
            if batch_all:
                return losses.sum()/(((losses > 1e-16).sum()).float()+1e-16), losses.mean()
            else:
                return losses.mean()
        else:
            return losses.sum()
    

if __name__ == "__main__":
    criterion = L2Loss(alpha=0.5, margin=0.2)
