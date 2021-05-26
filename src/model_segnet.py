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

from so3_pooling import SO3Pooling
from so3_unpooling import SO3Unpooling

def so3_to_s2_integrate(x):
    """
    Integrate a signal on SO(3) using the Haar measure

    :param x: [..., beta, alpha, gamma] (..., 2b, 2b, 2b)
    :return y: [...] (...)
    """
    assert x.size(-1) == x.size(-2)
    assert x.size(-2) == x.size(-3)
    return torch.sum(x, dim=-1) * (2*np.pi/x.size(-1))

class ModelSegnet(nn.Module):
    def __init__(self, bandwidth=100, n_classes=32):
        super().__init__()

        self.features = [2, 32, 64, 96, n_classes]        
        self.bandwidths = [bandwidth, 60, 30, 10, 30, 60, bandwidth] 

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/160, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 32, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        
        self.conv1 = nn.Sequential(
            S2Convolution(
                nfeature_in  = self.features[0],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[0],
                b_out = self.bandwidths[1],
                b_inverse = self.bandwidths[1],
                grid=grid_s2),
            nn.BatchNorm3d(self.features[1], affine=True),
            nn.PReLU()
        )
        
        self.max_pool1 = SO3Pooling(self.bandwidths[1], self.bandwidths[2])
        
        self.conv2 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[2],
                b_in  = self.bandwidths[2],
                b_out = self.bandwidths[2],
                b_inverse = self.bandwidths[2],
                grid=grid_so3_1),
            nn.BatchNorm3d(self.features[2], affine=True),
            nn.PReLU()
        )
        
        self.max_pool2 = SO3Pooling(self.bandwidths[2], self.bandwidths[3])
        
        self.unpool1 = SO3Unpooling(self.bandwidths[3], self.bandwidths[4])
        
        self.deconv1 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.features[2],
                nfeature_out = self.features[3],
                b_in  = self.bandwidths[4],
                b_out = self.bandwidths[4],
                b_inverse = self.bandwidths[4],
                grid=grid_so3_2),
            nn.BatchNorm3d(self.features[3], affine=True),
            nn.PReLU()
        )
        
        self.unpool2 = SO3Unpooling(self.bandwidths[4], self.bandwidths[5])
        
        self.deconv2 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.features[3],
                nfeature_out = self.features[4],
                b_in  = self.bandwidths[5],
                b_out = self.bandwidths[5],
                b_inverse = self.bandwidths[6],
                grid=grid_so3_3),
            nn.BatchNorm3d(self.features[4], affine=True),
            nn.PReLU()
        )
        
        self.sm = nn.LogSoftmax(dim=1)
        

    def forward(self, x):
        e1 = self.max_pool1(self.conv1(x))
        e2 = self.max_pool2(self.conv2(e1))
        
        d1 = self.deconv1(self.unpool1(e2))
        d2 = self.deconv2(self.unpool2(d1))
        
        return self.sm(so3_to_s2_integrate(d2))