import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

from lovasz_losses import *

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
#         weights = np.array([0., 0.16366589, 0.157578, 0.1621299, 0.08124114, 0.12953149, 0.16657334, 0.13928024, 0.])
        weights = np.array([0.03203128, 0.12453853, 0.12360233, 0.12430233, 0.1118631,  0.11928928, 0.12498565, 0.12078846, 0.11859904])
#         weights = np.array([1.00000e+00, 2.01360e+02, 6.64800e+01, 1.33180e+02, 7.07000e+00, 1.62700e+01, 6.47408e+03, 2.18100e+01, 1.45200e+01])
        assert(self.n_classes == len(weights))

        self.weights = torch.from_numpy(weights).cuda().float()

    def forward(self, decoded, teacher, size_average=True, batch_all=True):
        losses = F.nll_loss(decoded, teacher, weight=self.weights, size_average=size_average)

        if size_average:
            if batch_all:
                return losses.sum()/(((losses > 1e-16).sum()).float()+1e-16), losses.mean()
            else:
                return losses.mean()
        else:
            return losses.sum()

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        class_weights = np.array([0.03203128, 0.12453853, 0.12360233, 0.12430233, 0.1118631,  0.11928928, 0.12498565, 0.12078846, 0.11859904])
        super().__init__()
        self.log_softmax = nn.LogSoftmax()
        self.class_weights = autograd.Variable(torch.FloatTensor(class_weights).cuda())

    def forward(self, logits, target, size_average=True, batch_all=True):
        log_probabilities = self.log_softmax(logits)
        losses = -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()
        if size_average:
            if batch_all:
                return losses.sum()/(((losses > 1e-16).sum()).float()+1e-16), losses.mean()
            else:
                return losses.mean()
        else:
            return losses.sum()

def tv_loss(decoded, teacher):
#     print(f'decoded shape {decoded.shape} and teacher shape {teacher.shape}')
    n_batch, n_classes, n_height, n_width = decoded.shape
    decoder_w_variance = torch.sum(torch.pow(decoded[:,:,:,:-1] - decoded[:,:,:,1:], 2))
    decoder_h_variance = torch.sum(torch.pow(decoded[:,:,:-1,:] - decoded[:,:,1:,:], 2))
#     teacher_w_variance = torch.sum(torch.pow(teacher[:,:,:-1] - teacher[:,:,1:], 2))
#     teacher_h_variance = torch.sum(torch.pow(teacher[:,:-1,:] - teacher[:,1:,:], 2))
    return (decoder_h_variance + decoder_w_variance) / (n_batch * n_classes * n_height * n_width)

class MainLoss(nn.Module):
    def __init__(self):
        self.class_weights = np.array([0.03203128, 0.12453853, 0.12360233, 0.12430233, 0.1118631,  0.11928928, 0.12498565, 0.12078846, 0.11859904])
#         self.class_weights = np.array([0.0625, 0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625])
        super().__init__()
#         self.class_weights = autograd.Variable(torch.FloatTensor(self.class_weights).cuda())
        self.class_weights = torch.from_numpy(self.class_weights).cuda(2).float()
        self.alpha = 1.1
        self.beta = 1.6
        self.gamma = 7.0

    def forward(self, decoded, teacher, size_average=True, batch_all=True):
#         wce_losses = F.cross_entropy(decoded, teacher, weight=self.class_weights, size_average=size_average, ignore_index=250)
        nll_losses = F.nll_loss(F.log_softmax(decoded, dim=1), teacher, weight=self.class_weights, size_average=size_average)
        lz_losses = lovasz_softmax(F.softmax(decoded, dim=1), teacher)
#         tv_losses = tv_loss(decoded, teacher)

#         losses = self.alpha * nll_losses + self.beta * lz_losses + self.gamma * tv_losses
        losses = self.alpha * nll_losses + self.beta * lz_losses
        if size_average:
            if batch_all:
                return losses.sum()/(((losses > 1e-16).sum()).float()+1e-16), losses.mean()
            else:
                return losses.mean()
        else:
            return losses.sum()

if __name__ == "__main__":
    criterion = L2Loss(alpha=0.5, margin=0.2)
