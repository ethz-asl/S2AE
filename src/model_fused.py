
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

class ImageEncoder(nn.Module):
    def __init__(self, bandwidth, n_classes):
        super().__init__()

        self.features = [3, 32, 64]
        self.bandwidths = [bandwidth, 40, 10]

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/256, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/128, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        
        self.conv1 = nn.Sequential(
            S2Convolution(
                nfeature_in  = self.features[0],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[0],
                b_out = self.bandwidths[1],
                b_inverse = self.bandwidths[1],
                grid=grid_s2),
            nn.BatchNorm3d(self.features[1], affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[1],
                b_out = self.bandwidths[1],
                b_inverse = self.bandwidths[1],
                grid=grid_so3_1),
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
                grid=grid_so3_2),
            nn.BatchNorm3d(self.features[2], affine=True),
            nn.PReLU(),
        )
        
    def forward(self, x):
        # Encoder
#         print(f'fooooooooooo1 {x.size(0)} {x.size(1)} {x.size(2)}')
        e1 = self.conv1(x)
#         print(f'fooooooooooo2')
        mp = self.max_pool1(e1)
#         print(f'fooooooooooo3')
        e2 = self.conv2(mp)
#         print(f'fooooooooooo4')
        
#         return so3_to_s2_integrate(e2)
        return e2

class LidarEncoder(nn.Module):
    def __init__(self, bandwidth=100, n_classes=32):
        super().__init__()

        self.features = [n_classes, 32, 64]
        self.bandwidths = [bandwidth, 40, 10]

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/256, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/128, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        
        self.conv1 = nn.Sequential(
            S2Convolution(
                nfeature_in  = self.features[0],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[0],
                b_out = self.bandwidths[1],
                b_inverse = self.bandwidths[1],
                grid=grid_s2),
            nn.BatchNorm3d(self.features[1], affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[1],
                b_out = self.bandwidths[1],
                b_inverse = self.bandwidths[1],
                grid=grid_so3_1),
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
                grid=grid_so3_2),
            nn.BatchNorm3d(self.features[2], affine=True),
            nn.PReLU(),
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)
        e2 = self.conv2(self.max_pool1(e1))
        
#         return so3_to_s2_integrate(e2)
        return e2

class FusedDecoder(nn.Module):
    def __init__(self, bandwidth=10, n_classes=32):
        super().__init__()

        self.features = [64, 32, n_classes]
        self.bandwidths = [10, 40, 100]

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/256, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/128, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        
        self.deconv1 = nn.Sequential(
            nn.Dropout(p=0.1),
            SO3Convolution(
                nfeature_in  = self.features[0],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[0],
                b_out = self.bandwidths[0],
                b_inverse = self.bandwidths[0],
                grid=grid_so3_2),
            nn.BatchNorm3d(self.features[1], affine=True),
        )
        
        self.unpool1 = SO3Unpooling(self.bandwidths[0], self.bandwidths[1]) # 10 to 15 bw
        
        self.deconv2 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[1],
                b_out = self.bandwidths[1],
                b_inverse = self.bandwidths[1],
                grid=grid_so3_1),
            nn.BatchNorm3d(self.features[1], affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[2],
                b_in  = self.bandwidths[1],
                b_out = self.bandwidths[1],
                b_inverse = self.bandwidths[2],
                grid=grid_so3_1),
            nn.BatchNorm3d(self.features[2], affine=True),
            nn.PReLU()
        )
        
        self.lsm = nn.LogSoftmax(dim=1)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        d1 = self.deconv1(x)
        d2 = self.deconv2(self.unpool1(d1))
        
        # return self.sm(so3_to_s2_integrate(d4))
        return so3_to_s2_integrate(d2)


class FusedModel(nn.Module):
    def __init__(self, bandwidth=200, n_classes=32):
        super().__init__()
        self.lidar_encoder = LidarEncoder(100, n_classes).cuda()
        self.image_encoder = ImageEncoder(150, n_classes).cuda()
        self.fused_decoder = FusedDecoder(10, 16).cuda()
        
    def fuse_by_sum(self, e_lidar, e_img):
        return e_lidar + e_img

    def forward(self, x_lidar, x_img):
        e_lidar = self.lidar_encoder(x_lidar)
        e_img = self.image_encoder(x_img)
        fused = self.fuse_by_sum(e_lidar, e_img)
        return self.fused_decoder(fused)