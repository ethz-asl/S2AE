import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from s2cnn import so3_near_identity_grid, S2Convolution, s2_near_identity_grid, SO3Convolution, so3_integrate

class ModelEncodeDecodeSimple(nn.Module):
    def __init__(self, bandwidth=30):
        super().__init__()

        self.features = [2, 5]
        self.bandwidths = [bandwidth, 10]

        assert len(self.bandwidths) == len(self.features)
        grid_s2 = s2_near_identity_grid(n_alpha=6, max_beta=np.pi/160, n_beta=1)

        self.convolutional = nn.Sequential(
            S2Convolution(
                nfeature_in  = self.features[0],
                nfeature_out = self.features[1],
                b_in  = self.bandwidths[0],
                b_out = self.bandwidths[1],
                grid=grid_s2)
            )

    def forward(self, x1):
        x1 = self.convolutional(x1)  # [batch, feature, beta, alpha, gamma]
        return x1
