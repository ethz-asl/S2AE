import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from s2cnn import s2_near_identity_grid, so3_near_identity_grid
from s2cnn import so3_integrate

#from s2cnn import S2Convolution, SO3Convolution


from s2_conv import S2Convolution
from so3_conv import SO3Convolution

# pylint: disable=R,C,E1101
import torch
from functools import lru_cache
from s2cnn.utils.decorator import show_running


def so3_to_s2_integrate(x):
    """
    Integrate a signal on SO(3) using the Haar measure

    :param x: [..., beta, alpha, gamma] (..., 2b, 2b, 2b)
    :return y: [...] (...)
    """
    assert x.size(-1) == x.size(-2)
    assert x.size(-2) == x.size(-3)
    return torch.sum(x, dim=-1) * (2*np.pi/x.size(-1))

class ModelEncodeDecodeSimple(nn.Module):
    def __init__(self, bandwidth=100, n_classes=32):
        super().__init__()

        self.features = [2, 10, 20, 60, 80, 100, 120, n_classes]        
        self.bandwidths = [bandwidth, 70, 50, 30, 25, 15] 

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/160, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 32, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 24, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_5 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_6 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)


        self.convolutional = nn.Sequential(
            S2Convolution(
                nfeature_in  = self.features[0],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[0],
                b_out = self.bandwidths[1],
                b_inverse = self.bandwidths[1],
                grid=grid_s2),
            nn.PReLU(),
            nn.BatchNorm3d(self.features[1], affine=True),
            SO3Convolution(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[2],
                b_in  = self.bandwidths[1],
                b_out = self.bandwidths[2],
                b_inverse = self.bandwidths[2],
                grid=grid_so3_1),
            nn.PReLU(),
            nn.BatchNorm3d(self.features[2], affine=True),
            SO3Convolution(
                nfeature_in  = self.features[2],
                nfeature_out = self.features[3],
                b_in  = self.bandwidths[2],
                b_out = self.bandwidths[3],
                b_inverse = self.bandwidths[3],
                grid=grid_so3_2),
            nn.PReLU(),
            nn.BatchNorm3d(self.features[3], affine=True),
            SO3Convolution(
                nfeature_in  = self.features[3],
                nfeature_out = self.features[4],
                b_in  = self.bandwidths[3],
                b_out = self.bandwidths[4],
                b_inverse = self.bandwidths[4],
                grid=grid_so3_3),
            nn.PReLU(),
            nn.BatchNorm3d(self.features[4], affine=True),
            SO3Convolution(
                nfeature_in  = self.features[4],
                nfeature_out = self.features[5],
                b_in  = self.bandwidths[4],
                b_out = self.bandwidths[5],
                b_inverse = self.bandwidths[5],
                grid=grid_so3_4),
            nn.PReLU(),
            nn.BatchNorm3d(self.features[5], affine=True),
        )
        
        self.deconvolutional = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.features[5],
                nfeature_out = self.features[6],
                b_in  = self.bandwidths[5],
                b_out = self.bandwidths[5],
                b_inverse = self.bandwidths[5],
                grid=grid_so3_5),
            nn.PReLU(),
            nn.BatchNorm3d(self.features[6], affine=True),
            nn.Dropout(p=0.3),
            SO3Convolution(
                nfeature_in  = self.features[6],
                nfeature_out = self.features[7],
                b_in  = self.bandwidths[5],
                b_out = self.bandwidths[5],
                b_inverse = self.bandwidths[0],
                grid=grid_so3_5)            
        )
        
        self.sm = nn.LogSoftmax(dim=1)
        

    def forward(self, x1):
        #return self.sm(self.deconvolutional(self.convolutional(x1)).max(-1)[0])
        return self.sm(so3_to_s2_integrate(self.deconvolutional(self.convolutional(x1))))