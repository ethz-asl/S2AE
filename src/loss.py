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
#         self.class_weights = np.array([0.03203128, 0.12453853, 0.12360233, 0.12430233, 0.1118631,  0.11928928, 0.12498565, 0.12078846, 0.11859904])
        self.class_weights = np.array([0.0232466, 0.06180463, 0.06162999, 0.06086704, 0.06031089, 0.0611107, 0.06121712, 0.06137207, 0.06161231, 0.06170083, 0.06170534, 0.0586652, 0.06187166, 0.06135297, 0.06109214, 0.05989889, 0.06054162])

        super().__init__()
        self.class_weights = torch.from_numpy(self.class_weights).cuda().float()
        self.alpha = 1.1
        self.beta = 1.6
        self.gamma = 7.0

    def forward(self, decoded, teacher, size_average=True, batch_all=True):
#         wce_losses = F.cross_entropy(decoded, teacher, weight=self.class_weights, size_average=size_average, ignore_index=250)
        nll_losses = F.nll_loss(F.log_softmax(decoded, dim=1), teacher, weight=self.class_weights, size_average=size_average)
        # nll_losses = F.nll_loss(F.log_softmax(decoded, dim=1).cuda(2), teacher, size_average=size_average).cuda()
        lz_losses = lovasz_softmax(F.softmax(decoded, dim=1), teacher)
#         tv_losses = tv_loss(decoded, teacher)

#         losses = self.alpha * nll_losses + self.beta * lz_losses + self.gamma * tv_losses
        losses = self.alpha * nll_losses + self.beta * lz_losses
        return losses

class WceLovasz(nn.Module):
    def __init__(self, ignore = 0):
        super().__init__()
        self.ignore = ignore
#         self.class_weights = np.array([0, 1.58770358, 4.08255301, 1., 1.26586145, 1.11405485, 1.25146044])
#         self.class_weights = np.array([0, 1.50119213, 4.32795156, 1.0, 1.14440329, 1.07012146, 1.09708853]) # new weights
#         self.class_weights = np.array([0.0, 108.11283694, 2576.5262789, 15.96423925, 44.74384341, 45.75927258])
        # self.class_weights = np.array([0.0, 110.80529464, 2096.92746278, 18.74137809, 57.46342281, 46.27912271])
        self.class_weights = np.array([0.0, 114.40187275, 24736.35886674, 14.54492561, 38.49370199, 30.57194287])
        self.class_weights = torch.from_numpy(self.class_weights).cuda().float()
        self.wce = torch.nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=self.ignore)

    def forward(self, decoded, teacher, size_average=True, batch_all=True):
        wce_losses = self.wce(decoded, teacher)
        lz_losses = lovasz_softmax(F.softmax(decoded, dim=1), teacher, ignore=self.ignore)

        return wce_losses + lz_losses

if __name__ == "__main__":
    criterion = L2Loss(alpha=0.5, margin=0.2)
