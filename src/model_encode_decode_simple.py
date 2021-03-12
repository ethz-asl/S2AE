import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from s2cnn import s2_near_identity_grid
from s2_deconv import S2Deconvolution
from s2_conv import S2Convolution

# pylint: disable=R,C,E1101
import torch
from functools import lru_cache
from s2cnn.utils.decorator import show_running


def so3_integrate(x):
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
    def __init__(self, bandwidth=30):
        super().__init__()

        self.features = [2, 10]
        self.bandwidths = [bandwidth, 30]

        assert len(self.bandwidths) == len(self.features)
        grid_s2 = s2_near_identity_grid(n_alpha=6, max_beta=np.pi/160, n_beta=1)

        self.convolutional = S2Convolution(
                nfeature_in  = self.features[0],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[0],
                b_out = self.bandwidths[1],
                grid=grid_s2)
        self.deconvolutional = S2Deconvolution(
                nfeature_in  = self.features[1],
                nfeature_out = self.features[0],
                b_in  = self.bandwidths[1],
                b_out = self.bandwidths[1],
                grid=grid_s2)

    def forward(self, x1):
        x_enc = self.convolutional(x1)  # [batch, feature, beta, alpha, gamma]
        print(f"encoded x shape is {x_enc.shape}")
        x_enc = so3_integrate(x_enc)  # [batch, feature]
        print(f"integrated x shape is {x_enc.shape}")
        x_dec = self.deconvolutional(x_enc)  # [batch, feature, beta, alpha, gamma]
        x_dec = so3_integrate(x_dec)  # [batch, feature]
        return x_dec
