
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


class Normalize(torch.nn.Module):

    def __init__(self, mu, sigma):
        super().__init__()

        self.register_buffer('mu', torch.FloatTensor([mu]))
        self.register_buffer('sigma', torch.FloatTensor([sigma]))

    def forward(self, x):
        normalized = ((x - self.mu[None, :, None, None].to(x.device)) /
                      self.sigma[None, :, None, None].to(x.device))
        return normalized


class Model(nn.Module):
    def __init__(self, bandwidth=100, n_classes=32):
        super().__init__()

        mu = 1.5984423678560475
        sigma = 8.400141767886875
        self.normalize = Normalize(mu, sigma)

        # TESTING
        self.features = [1, 30, 45, 120, 200, 120, 45, 30, n_classes]
        self.bandwidths = [bandwidth, 40, 30,
                           20, 10, 8, 10, 20, 30, 40, bandwidth]

        # TESTING cluster
        self.features = [1, 40, 50, 120, 200, 120, 50, 40, n_classes]
        self.bandwidths = [bandwidth, 40, 30,
                           20, 10, 8, 10, 20, 30, 40, bandwidth]

        # v3 with 7 classes with bw120 (small)
#         self.features = [1, 15, 40, 70, 100, 70, 40, 15, n_classes]
#         self.bandwidths = [bandwidth, 40, 30, 15, 10, 8, 10, 15, 30, 40, bandwidth]

        # v1 with 9
        # self.features = [2, 20, 40, 90, 160, 90, 40, 20, n_classes]
        # self.bandwidths = [bandwidth, 40, 30, 15, 10, 8, 10, 15, 30, 40, bandwidth]

        # v1 with 7
        #self.features = [2, 20, 45, 140, 180, 140, 45, 20, n_classes]
        #self.bandwidths = [bandwidth, 40, 30, 15, 10, 8, 10, 15, 30, 40, bandwidth]

#         small model 5x11GB
        #self.features = [2, 20, 40, 60, 100, 60, 40, 20, n_classes]
        #self.bandwidths = [bandwidth, 50, 30, 15, 10, 8, 10, 15, 30, 50, bandwidth]

        # big model 5x24GB (batch size = 5)
        #self.features = [2, 20, 40, 60, 120, 60, 40, 20, n_classes]
        #self.bandwidths = [bandwidth, 60, 40, 30, 20, 10, 20, 30, 40, 60, bandwidth]

        # big model 5x24GB (batch size = 10)
#         self.features = [2, 20, 40, 80, 180, 80, 40, 20, n_classes]
#         self.bandwidths = [bandwidth, 50, 40, 20, 15, 10, 15, 20, 40, 50, bandwidth]

        print(f'[Model] We have {self.features} features.')
        print(f'[Model] We have {self.bandwidths} bandwidths.')

        grid_size = 8
        grid_s2 = s2_near_identity_grid(
            n_alpha=grid_size, max_beta=np.pi/16, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(
            n_alpha=grid_size, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=grid_size)
        grid_so3_2 = so3_near_identity_grid(
            n_alpha=grid_size, max_beta=np.pi / 8, n_beta=1, max_gamma=2*np.pi, n_gamma=grid_size)
        grid_so3_3 = so3_near_identity_grid(
            n_alpha=grid_size, max_beta=np.pi / 4, n_beta=1, max_gamma=2*np.pi, n_gamma=grid_size)
        grid_so3_4 = so3_near_identity_grid(
            n_alpha=grid_size, max_beta=np.pi / 2, n_beta=1, max_gamma=2*np.pi, n_gamma=grid_size)

        self.conv1 = nn.Sequential(
            S2Convolution(
                nfeature_in=self.features[0],
                nfeature_out=self.features[1],
                b_in=self.bandwidths[0],
                b_out=self.bandwidths[1],
                b_inverse=self.bandwidths[1],
                grid=grid_s2),
            nn.BatchNorm3d(self.features[1], affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in=self.features[1],
                nfeature_out=self.features[1],
                b_in=self.bandwidths[1],
                b_out=self.bandwidths[1],
                b_inverse=self.bandwidths[1],
                grid=grid_so3_1),
            nn.BatchNorm3d(self.features[1], affine=True),
            nn.PReLU()
        )

        self.max_pool1 = SO3Pooling(self.bandwidths[1], self.bandwidths[2])

        self.conv2 = nn.Sequential(
            SO3Convolution(
                nfeature_in=self.features[1],
                nfeature_out=self.features[2],
                b_in=self.bandwidths[2],
                b_out=self.bandwidths[2],
                b_inverse=self.bandwidths[2],
                grid=grid_so3_2),
            nn.BatchNorm3d(self.features[2], affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in=self.features[2],
                nfeature_out=self.features[2],
                b_in=self.bandwidths[2],
                b_out=self.bandwidths[2],
                b_inverse=self.bandwidths[2],
                grid=grid_so3_2),
            nn.BatchNorm3d(self.features[2], affine=True),
            nn.PReLU(),
            nn.Dropout3d(p=0.05),
        )

        self.max_pool2 = SO3Pooling(self.bandwidths[2], self.bandwidths[3])

        self.conv3 = nn.Sequential(
            SO3Convolution(
                nfeature_in=self.features[2],
                nfeature_out=self.features[3],
                b_in=self.bandwidths[3],
                b_out=self.bandwidths[3],
                b_inverse=self.bandwidths[3],
                grid=grid_so3_3),
            nn.BatchNorm3d(self.features[3], affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in=self.features[3],
                nfeature_out=self.features[3],
                b_in=self.bandwidths[3],
                b_out=self.bandwidths[3],
                b_inverse=self.bandwidths[3],
                grid=grid_so3_3),
            nn.BatchNorm3d(self.features[3], affine=True),
            nn.PReLU(),
            nn.Dropout3d(p=0.05),
        )

        self.max_pool3 = SO3Pooling(self.bandwidths[3], self.bandwidths[4])

        self.conv4 = nn.Sequential(
            SO3Convolution(
                nfeature_in=self.features[3],
                nfeature_out=self.features[4],
                b_in=self.bandwidths[4],
                b_out=self.bandwidths[4],
                b_inverse=self.bandwidths[4],
                grid=grid_so3_4),
            nn.BatchNorm3d(self.features[4], affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in=self.features[4],
                nfeature_out=self.features[4],
                b_in=self.bandwidths[4],
                b_out=self.bandwidths[4],
                b_inverse=self.bandwidths[4],
                grid=grid_so3_4),
            nn.BatchNorm3d(self.features[4], affine=True),
            nn.PReLU(),
            nn.Dropout3d(p=0.05),
        )

        # -------------------------------------------------------------------------------

        self.deconv1 = nn.Sequential(
            SO3Convolution(
                nfeature_in=self.features[4],
                nfeature_out=self.features[4],
                b_in=self.bandwidths[6],
                b_out=self.bandwidths[6],
                b_inverse=self.bandwidths[6],
                grid=grid_so3_4),
            nn.BatchNorm3d(self.features[4], affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in=self.features[4],
                nfeature_out=self.features[5],
                b_in=self.bandwidths[6],
                b_out=self.bandwidths[6],
                b_inverse=self.bandwidths[6],
                grid=grid_so3_4),
            nn.BatchNorm3d(self.features[5], affine=True),
            nn.PReLU()
        )

        self.unpool2 = SO3Unpooling(
            self.bandwidths[6], self.bandwidths[7])  # 10 to 15 bw

        self.deconv2 = nn.Sequential(
            SO3Convolution(
                nfeature_in=self.features[5] + self.features[3],
                nfeature_out=self.features[5],
                b_in=self.bandwidths[7],
                b_out=self.bandwidths[7],
                b_inverse=self.bandwidths[7],
                grid=grid_so3_3),
            nn.BatchNorm3d(self.features[5], affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in=self.features[5],
                nfeature_out=self.features[6],
                b_in=self.bandwidths[7],
                b_out=self.bandwidths[7],
                b_inverse=self.bandwidths[7],
                grid=grid_so3_3),
            nn.BatchNorm3d(self.features[6], affine=True),
            nn.PReLU()
        )

        self.unpool3 = SO3Unpooling(self.bandwidths[7], self.bandwidths[8])

        self.deconv3 = nn.Sequential(
            SO3Convolution(
                nfeature_in=self.features[6] + self.features[2],
                nfeature_out=self.features[6],
                b_in=self.bandwidths[8],
                b_out=self.bandwidths[8],
                b_inverse=self.bandwidths[8],
                grid=grid_so3_2),
            nn.BatchNorm3d(self.features[6], affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in=self.features[6],
                nfeature_out=self.features[7],
                b_in=self.bandwidths[8],
                b_out=self.bandwidths[8],
                b_inverse=self.bandwidths[8],
                grid=grid_so3_2),
            nn.BatchNorm3d(self.features[7], affine=True),
            nn.PReLU()
        )

        self.unpool4 = SO3Unpooling(self.bandwidths[8], self.bandwidths[9])

        self.deconv4 = nn.Sequential(
            SO3Convolution(
                nfeature_in=self.features[7] + self.features[1],
                nfeature_out=self.features[7],
                b_in=self.bandwidths[9],
                b_out=self.bandwidths[9],
                b_inverse=self.bandwidths[9],
                grid=grid_so3_1),
            nn.BatchNorm3d(self.features[7], affine=True),
            nn.PReLU(),
            SO3Convolution(
                nfeature_in=self.features[7],
                nfeature_out=self.features[8],
                b_in=self.bandwidths[9],
                b_out=self.bandwidths[9],
                b_inverse=self.bandwidths[10],
                grid=grid_so3_1),
        )


#         self.lsm = nn.LogSoftmax(dim=1)
#         self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.normalize(x)

        # Encoder
        e1 = self.conv1(x)
        e2 = self.conv2(self.max_pool1(e1))
        e3 = self.conv3(self.max_pool2(e2))
        e4 = self.conv4(self.max_pool3(e3))

        # Decoder
        d1 = self.deconv1(e4)

        ud1 = self.unpool2(d1)
        e3d1 = torch.cat([e3, ud1], dim=1)
        d2 = self.deconv2(e3d1)

        e2d2 = torch.cat([e2, self.unpool3(d2)], dim=1)
        d3 = self.deconv3(e2d2)

        e1d3 = torch.cat([e1, self.unpool4(d3)], dim=1)
        d4 = self.deconv4(e1d3)

        # return self.sm(so3_to_s2_integrate(d4))
        return so3_to_s2_integrate(d4)
