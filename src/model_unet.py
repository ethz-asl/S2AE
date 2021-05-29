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

class ModelUnet(nn.Module):
    def __init__(self, bandwidth=100, n_classes=32):
        super().__init__()

        self.left_features = [2, 60, 80, 100]
        self.bottom_features = [100, 150, 100]
        self.right_features = [200, 80, 160, 60, 120, n_classes]
        self.bandwidths = [bandwidth, 35, 25, 10, 25, 35, bandwidth]

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi / 64, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi / 64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi / 32, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi / 24, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi / 24, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        
        # --- Left layers ------------------------------------------------------------
        
        self.left_conv1 = nn.Sequential(
            S2Convolution(
                nfeature_in  = self.left_features[0],
                nfeature_out = self.left_features[1],
                b_in  = self.bandwidths[0],
                b_out = self.bandwidths[1],
                b_inverse = self.bandwidths[1],
                grid=grid_s2),
            nn.PReLU(),
            nn.BatchNorm3d(self.left_features[1], affine=True)
        )
        
        self.left_conv2 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.left_features[1],
                nfeature_out = self.left_features[2],
                b_in  = self.bandwidths[1],
                b_out = self.bandwidths[2],
                b_inverse = self.bandwidths[2],
                grid=grid_so3_1),
            nn.PReLU(),
            nn.BatchNorm3d(self.left_features[2], affine=True),
        )
        
        self.left_conv3 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.left_features[2],
                nfeature_out = self.left_features[3],
                b_in  = self.bandwidths[2],
                b_out = self.bandwidths[3],
                b_inverse = self.bandwidths[3],
                grid=grid_so3_2),
            nn.PReLU(),
            nn.BatchNorm3d(self.left_features[3], affine=True),
        )
        
        # --- Bottom layers ------------------------------------------------------------
        
        self.bottom_conv1 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.bottom_features[0],
                nfeature_out = self.bottom_features[1],
                b_in  = self.bandwidths[3],
                b_out = self.bandwidths[3],
                b_inverse = self.bandwidths[3],
                grid=grid_so3_3),
            nn.PReLU(),
            nn.BatchNorm3d(self.bottom_features[1], affine=True),
        )
        
        self.bottom_conv2 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.bottom_features[1],
                nfeature_out = self.bottom_features[2],
                b_in  = self.bandwidths[3],
                b_out = self.bandwidths[3],
                b_inverse = self.bandwidths[3],
                grid=grid_so3_3),
            nn.PReLU(),
            nn.BatchNorm3d(self.bottom_features[2], affine=True),
        )
        
        # --- Right layers ------------------------------------------------------------
        
        self.right_conv1 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.right_features[0],
                nfeature_out = self.right_features[1],
                b_in  = self.bandwidths[3],
                b_out = self.bandwidths[3],
                b_inverse = self.bandwidths[4],
                grid=grid_so3_2),
            nn.PReLU(),
            nn.BatchNorm3d(self.right_features[1], affine=True),
        )
        
        self.right_conv2 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.right_features[2],
                nfeature_out = self.right_features[3],
                b_in  = self.bandwidths[4],
                b_out = self.bandwidths[4],
                b_inverse = self.bandwidths[5],
                grid=grid_so3_1),
            nn.PReLU(),
            nn.BatchNorm3d(self.right_features[3], affine=True),
        )
        
        self.right_conv3 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.right_features[4],
                nfeature_out = self.right_features[5],
                b_in  = self.bandwidths[5],
                b_out = self.bandwidths[5],
                b_inverse = self.bandwidths[6],
                grid=grid_so3_4)            
        )
        
        self.sm = nn.LogSoftmax(dim=1)
        

    def forward(self, x):        
        l1 = self.left_conv1(x)
        l2 = self.left_conv2(l1)
        l3 = self.left_conv3(l2)
        
        b1 = self.bottom_conv1(l3)
        b2 = self.bottom_conv2(b1)
        
        l3b2 = torch.cat([l3, b2], dim=1)
        r1 = self.right_conv1(l3b2)
        
        l2r1 = torch.cat([l2, r1], dim=1)
        r2 = self.right_conv2(l2r1)
        
        l1r2 = torch.cat([l1, r2], dim=1)
        r3 = self.right_conv3(l1r2)
        
        return self.sm(so3_to_s2_integrate(r3))
