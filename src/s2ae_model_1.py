import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from s2cnn import s2_near_identity_grid
from s2_deconv import S2Deconvolution
from s2_conv import S2Convolution
from so3_pooling import SO3Pooling
from so3_unpooling import SO3Unpooling

# pylint: disable=R,C,E1101
import torch
from functools import lru_cache
from s2cnn.utils.decorator import show_running


class S2AE_Model_1(nn.Module):
    def __init__(self, bandwidth=30):
        super().__init__()

        self.features = [2, 10, 30]
        self.bandwidths = [bandwidth, 30, 10, 5]

        assert len(self.bandwidths) == len(self.features)
        grid_s2 = s2_near_identity_grid(n_alpha=6, max_beta=np.pi/160, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.convolutional = nn.Sequential(
            # First conv block
            S2Convolution(
                nfeature_in  = self.features[0],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[0],
                b_out = self.bandwidths[1],
                grid=grid_s2),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(self.features[1], affine=True),
            SO3Convolution(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[1],
                b_out = self.bandwidths[1],
                grid=grid_so3_1),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(self.features[1], affine=True),
            SO3Pooling(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[1],
                b_out = self.bandwidths[2]),

            # Second conv block
            SO3Convolution(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[2],
                b_in  = self.bandwidths[2],
                b_out = self.bandwidths[2],
                grid=grid_so3_2),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(self.features[2], affine=True),
            SO3Convolution(
                nfeature_in  = self.features[2],
                nfeature_out = self.features[2],
                b_in  = self.bandwidths[2],
                b_out = self.bandwidths[2],
                grid=grid_so3_2),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(self.features[2], affine=True),
            SO3Pooling(
                nfeature_in  = self.features[2],
                nfeature_out = self.features[2],
                b_in  = self.bandwidths[2],
                b_out = self.bandwidths[3]),
        )

        self.deconvolutional = nn.Sequential(
            # First deconv block
            SO3Unpooling(
                nfeature_in  = self.features[2],
                nfeature_out = self.features[2],
                b_in  = self.bandwidths[3],
                b_out = self.bandwidths[2]),
            SO3Convolution(
                nfeature_in  = self.features[2],
                nfeature_out = self.features[2],
                b_in  = self.bandwidths[2],
                b_out = self.bandwidths[2],
                grid=grid_so3_2),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(self.features[2], affine=True),
            SO3Convolution(
                nfeature_in  = self.features[2],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[2],
                b_out = self.bandwidths[2],
                grid=grid_so3_2),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(self.features[1], affine=True),

            # Second deconv block
            SO3Unpooling(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[2],
                b_out = self.bandwidths[1]),
            SO3Convolution(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[1],
                b_out = self.bandwidths[1],
                grid=grid_so3_1),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(self.features[1], affine=True),
        )

    def forward(self, x1):
        x_enc = self.convolutional(x1)  # [batch, feature, beta, alpha, gamma]
        print(f"encoded x shape is {x_enc.shape}")
        #x_enc = so3_integrate(x_enc)  # [batch, feature]
        #print(f"integrated x shape is {x_enc.shape}")
        return x_enc
        #x_dec = self.deconvolutional(x_enc)  # [batch, feature, beta, alpha, gamma]
        #x_dec = so3_integrate(x_dec)  # [batch, feature]
        #return x_dec
