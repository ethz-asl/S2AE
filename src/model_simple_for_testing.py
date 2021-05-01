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


class ModelSimpleForTesting(nn.Module):
    def __init__(self, bandwidth=100, n_classes=32):
        super().__init__()

        self.features = [2, 5, 10, n_classes]        
        self.bandwidths = [bandwidth, 7, 6] 

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/170, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/100, n_beta=1, max_gamma=2*np.pi, n_gamma=6)        


        self.convolutional = nn.Sequential(
            S2Convolution(
                nfeature_in  = self.features[0],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[0],
                b_out = self.bandwidths[1],
                b_inverse = self.bandwidths[1],
                grid=grid_s2),
            nn.ReLU(),
#             nn.BatchNorm3d(self.features[1], affine=True),
            SO3Convolution(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[2],
                b_in  = self.bandwidths[1],
                b_out = self.bandwidths[2],
                b_inverse = self.bandwidths[2],
                grid=grid_so3_1),
            nn.ReLU(),
#             nn.BatchNorm3d(self.features[2], affine=True),
        )
        
        self.deconvolutional = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.features[2],
                nfeature_out = self.features[3],
                b_in  = self.bandwidths[2],
                b_out = self.bandwidths[2],
                b_inverse = self.bandwidths[0],
                grid=grid_so3_1)            
        )
        
        self.sm = nn.LogSoftmax(dim=1)
        

    def forward(self, x1):
        return self.sm(self.deconvolutional(self.convolutional(x1)).max(-1)[0])
