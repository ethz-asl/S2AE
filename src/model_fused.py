
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

        self.features = [3, 30, 40]
        self.bandwidths = [bandwidth, 25, 5]

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/128, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/32, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        
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
        e1 = self.conv1(x)
#         mp = self.max_pool1(e1)
#         e2 = self.conv2(mp)
        e2 = self.conv2(self.max_pool1(e1))
        
#         return so3_to_s2_integrate(e2)
        return e1, e2

class LidarEncoder(nn.Module):
    def __init__(self, bandwidth=100, n_classes=32):
        super().__init__()

        self.features = [n_classes, 30, 40]
        self.bandwidths = [bandwidth, 25, 5]

#         grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/256, n_beta=1)
#         grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/128, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
#         grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        
        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/128, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/32, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        
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
        return e1, e2

class FusedDecoder(nn.Module):
    def __init__(self, bandwidth=10, n_classes=32):
        super().__init__()

        self.features = [80, 30, 9]
        self.bandwidths = [5, 25, 100]

#         grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/128, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
#         grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/32, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        
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
        
        self.skip_size = 30 + 30
        
        self.deconv2 = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.features[1] + self.skip_size,
                nfeature_out = self.features[1] + self.skip_size,
                b_in  = self.bandwidths[1],
                b_out = self.bandwidths[1],
                b_inverse = self.bandwidths[1],
                grid=grid_so3_1),
            nn.BatchNorm3d(self.features[1] + self.skip_size, affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in  = self.features[1] + self.skip_size,
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

    def forward(self, x, e_img_lidar):
        d1 = self.deconv1(x)
#         d2 = self.deconv2(self.unpool1(d1))

        # Skip connection
        ud1 = self.unpool1(d1)
        e_ud1 = torch.cat([e_img_lidar, ud1], dim=1)
        d2 = self.deconv2(e_ud1)
        
        # return self.sm(so3_to_s2_integrate(d4))
        return so3_to_s2_integrate(d2)


class FusedModel(nn.Module):
    def __init__(self, bandwidth=200, n_classes=32):
        super().__init__()
        self.lidar_encoder = LidarEncoder(100, n_classes).cuda()
        self.image_encoder = ImageEncoder(150, n_classes).cuda()
        self.fused_decoder = FusedDecoder(10, 16).cuda()
        
        self.feature_bw = 2*5
        self.features = 80
        self.map_to_so3 = nn.Sequential(
            nn.Linear(in_features=self.features,out_features=self.features*self.feature_bw,bias=False),
            nn.PReLU(),
            nn.Linear(in_features=self.features*self.feature_bw,out_features=self.features*(self.feature_bw**2),bias=False),
            nn.PReLU(),
            nn.Linear(in_features=self.features*(self.feature_bw**2),out_features=self.features*(self.feature_bw**3),bias=False),
            nn.PReLU()
        )
        
    def fuse_by_sum(self, e_lidar, e_img):
        return e_lidar + e_img
    
    def fuse_by_avg(self, e_lidar, e_img):
        return (e_lidar + e_img) / 2
    
    def fuse_by_concat(self, e_lidar, e_img):
        return torch.cat([e_lidar, e_img], dim=1)
    
    def fuse_by_fc(self, e_lidar, e_img):
        feature_lidar = so3_integrate(e_lidar) # from [batch, feature, alpha, beta, gamma] to [batch, feature]
        feature_image = so3_integrate(e_img) # from [batch, feature, alpha, beta, gamma] to [batch, feature]
        
        n_batch = feature_lidar.shape[0] 
        n_features = feature_lidar.shape[1] + feature_image.shape[1]
        feature = torch.cat([feature_lidar, feature_image], dim=1) # [batch, 2xfeature]
        
        feature = self.map_to_so3(feature) # from [batch, 2xfeature] to [batch, 2xfeature, alpha, beta, gamma]
        return torch.reshape(feature, (n_batch, n_features, self.feature_bw, self.feature_bw, self.feature_bw))
    
    def fuse(self, e_lidar, e_img):
#         return self.fuse_by_sum(e_lidar, e_img)
#         return self.fuse_by_avg(e_lidar, e_img)
#         return self.fuse_by_concat(e_lidar, e_img)
        return self.fuse_by_fc(e_lidar, e_img)

    def forward(self, x_lidar, x_img):
        e1_lidar, e2_lidar = self.lidar_encoder(x_lidar)
        e1_img, e2_img = self.image_encoder(x_img)
        fused = self.fuse(e2_lidar, e2_img)
        return self.fused_decoder(fused, torch.cat([e1_lidar, e1_img], dim=1))
        