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

    b = x.size(-1) // 2

    w = _setup_so3_integrate(b, device_type=x.device.type, device_index=x.device.index)  # [beta]

    #x = torch.sum(x, dim=-1).squeeze(-1)  # [..., beta, alpha]
    #x = torch.sum(x, dim=-1).squeeze(-1)  # [..., beta]

    #print(f"size of x: {x.size()},  w {w.size()}")

    sz = x.size()
    x = x.view(-1, 2 * b)

    w = w.view(2 * b, 1)
    x = torch.mm(x, w).squeeze(-1)
    x = x.view(*sz[:-1])
    return x


@lru_cache(maxsize=32)
@show_running
def _setup_so3_integrate(b, device_type, device_index):
    import lie_learn.spaces.S3 as S3

    return torch.tensor(S3.quadrature_weights(b), dtype=torch.float32, device=torch.device(device_type, device_index))  # (2b) [beta]  # pylint: disable=E1102

class ModelEncodeDecodeSimple(nn.Module):
    def __init__(self, bandwidth=100, n_classes=32):
        super().__init__()

        self.features = [2, 10, 20, 60, 100, n_classes]        
        self.bandwidths = [bandwidth, 70, 50, 40, 30] 

        grid_s2    =  s2_near_identity_grid(n_alpha=6, max_beta=np.pi/160, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/64, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 32, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 24, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)


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
        )
        
        self.deconvolutional = nn.Sequential(
            SO3Convolution(
                nfeature_in  = self.features[4],
                nfeature_out = self.features[5],
                b_in  = self.bandwidths[4],
                b_out = self.bandwidths[4],
                b_inverse = self.bandwidths[0],
                grid=grid_so3_4)            
        )
        

    def forward(self, x1):
        x_enc = self.convolutional(x1)  # [batch, feature, beta, alpha, gamma]
        x_dec = self.deconvolutional(x_enc)  # [batch, feature, beta, alpha, gamma]

        # integrate out the gamma component to get a signal on S^2.
        #x_dec = so3_to_s2_integrate(x_dec)  # [batch, feature]

        # Max pooling of the gamma component
        #x_dec = x_dec.max(-1)[0]
        print(f"shape of x_dec is {x_dec.shape}")
        #m = nn.MaxPool3d((x_dec.shape[0], x_dec.shape[1], 1), stride=(1,1,1))
        #res = m(x_dec)
        #res, _ = torch.max(x_dec, 1)
        res = x_dec.max(-1)[0]
        print(f"shape of max pooled is {res.shape}")
                    
        return res
