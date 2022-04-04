
import torch
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


class FusedModel(nn.Module):
    def __init__(self, bandwidth=200, n_classes=32):
        super().__init__()

        self.dev0 = 'cuda:0'
        self.dev1 = 'cuda:1'
        self.dev2 = 'cuda:2'

        # IMAGE:
        # ------------------------------------------------------------------------

        self.image_features = [3, 80, 150]
        self.image_bandwidths = [150, 20, 10]

        image_grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/128, n_beta=1)
        image_grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        image_grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/32, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.image_conv1 = nn.Sequential(
            S2Convolution(
                nfeature_in  = self.image_features[0],
                nfeature_out = self.image_features[1],
                b_in  = self.image_bandwidths[0],
                b_out = self.image_bandwidths[1],
                b_inverse = self.image_bandwidths[1],
                grid=image_grid_s2),
            nn.BatchNorm3d(self.image_features[1], affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in  = self.image_features[1],
                nfeature_out = self.image_features[1],
                b_in  = self.image_bandwidths[1],
                b_out = self.image_bandwidths[1],
                b_inverse = self.image_bandwidths[1],
                grid=image_grid_so3_1),
            nn.BatchNorm3d(self.image_features[1], affine=True),
            nn.PReLU()
        ).to(self.dev1)

        self.image_max_pool1 = SO3Pooling(self.image_bandwidths[1], self.image_bandwidths[2]).to(self.dev0)

        self.image_conv2 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.image_features[1],
                nfeature_out = self.image_features[2],
                b_in  = self.image_bandwidths[2],
                b_out = self.image_bandwidths[2],
                b_inverse = self.image_bandwidths[2],
                grid=image_grid_so3_2),
            nn.BatchNorm3d(self.image_features[2], affine=True),
            nn.PReLU(),
        ).to(self.dev0)

        # LiDAR:
        # ------------------------------------------------------------------------
        self.lidar_features = [9, 80, 150]
        self.lidar_bandwidths = [100, 20, 10]

        lidar_grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/128, n_beta=1)
        lidar_grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        lidar_grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/32, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.lidar_conv1 = nn.Sequential(
            S2Convolution(
                nfeature_in  = self.lidar_features[0],
                nfeature_out = self.lidar_features[1],
                b_in  = self.lidar_bandwidths[0],
                b_out = self.lidar_bandwidths[1],
                b_inverse = self.lidar_bandwidths[1],
                grid=lidar_grid_s2),
            nn.BatchNorm3d(self.lidar_features[1], affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in  = self.lidar_features[1],
                nfeature_out = self.lidar_features[1],
                b_in  = self.lidar_bandwidths[1],
                b_out = self.lidar_bandwidths[1],
                b_inverse = self.lidar_bandwidths[1],
                grid=lidar_grid_so3_1),
            nn.BatchNorm3d(self.lidar_features[1], affine=True),
            nn.PReLU()
        ).to(self.dev1)

        self.lidar_max_pool1 = SO3Pooling(self.lidar_bandwidths[1], self.lidar_bandwidths[2]).to(self.dev0)

        self.lidar_conv2 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.lidar_features[1],
                nfeature_out = self.lidar_features[2],
                b_in  = self.lidar_bandwidths[2],
                b_out = self.lidar_bandwidths[2],
                b_inverse = self.lidar_bandwidths[2],
                grid=lidar_grid_so3_2),
            nn.BatchNorm3d(self.lidar_features[2], affine=True),
            nn.PReLU(),
        ).to(self.dev0)

        # FUSED:
        # ------------------------------------------------------------------------

        self.fused_features = [150, 100, 9]
        self.fused_bandwidths = [10, 20, 100]

        fused_grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        fused_grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/32, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.deconv1 = nn.Sequential(
            nn.Dropout(p=0.1),
            SO3Convolution(
                nfeature_in  = self.fused_features[0],
                nfeature_out = self.fused_features[1],
                b_in  = self.fused_bandwidths[0],
                b_out = self.fused_bandwidths[0],
                b_inverse = self.fused_bandwidths[0],
                grid=fused_grid_so3_2),
            nn.BatchNorm3d(self.fused_features[1], affine=True),
        ).to(self.dev0)

        self.unpool1 = SO3Unpooling(self.fused_bandwidths[0], self.fused_bandwidths[1]).to(self.dev0)

        self.skip_size = 0
        self.deconv2 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.fused_features[1] + self.skip_size,
                nfeature_out = self.fused_features[1] + self.skip_size,
                b_in  = self.fused_bandwidths[1],
                b_out = self.fused_bandwidths[1],
                b_inverse = self.fused_bandwidths[1],
                grid=fused_grid_so3_1),
            nn.BatchNorm3d(self.fused_features[1] + self.skip_size, affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in  = self.fused_features[1] + self.skip_size,
                nfeature_out = self.fused_features[2],
                b_in  = self.fused_bandwidths[1],
                b_out = self.fused_bandwidths[1],
                b_inverse = self.fused_bandwidths[2],
                grid=fused_grid_so3_1),
            nn.BatchNorm3d(self.fused_features[2], affine=True),
            nn.PReLU()
        ).to(self.dev2)

    # ------------------------------------------------------------------------

    def fuse_by_sum(self, e_lidar, e_img):
        return e_lidar + e_img

    def fuse_by_avg(self, e_lidar, e_img):
        return (e_lidar + e_img) / 2

    def fuse_by_concat(self, e_lidar, e_img):
        return torch.cat([e_lidar, e_img], dim=1)

    def fuse(self, e_lidar, e_img):
        return self.fuse_by_sum(e_lidar, e_img)
#         return self.fuse_by_avg(e_lidar, e_img)
        # return self.fuse_by_concat(e_lidar, e_img)

    def forward(self, x_lidar, x_img):

        # Image encoder:
        e1_img = self.image_conv1(x_img.to(self.dev1)).to(self.dev0)
        e2_img = self.image_conv2(self.image_max_pool1(e1_img))
        print(f'Finished image encoding')

        # LiDAR encoder:
        e1_lidar = self.lidar_conv1(x_lidar.to(self.dev1)).to(self.dev0)
        e2_lidar = self.lidar_conv2(self.lidar_max_pool1(e1_lidar))
        print(f'Finished lidar encoding')

        fused = self.fuse(e2_lidar, e2_img)
        d1 = self.deconv1(x)
        ud1 = self.unpool1(d1)
        d2 = self.deconv2(ud1.to(self.dev2))
        dec = so3_to_s2_integrate(d2)
        print(f'Finished decoding')

        return dec
