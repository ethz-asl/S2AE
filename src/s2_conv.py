# from: https://github.com/jonkhler/s2cnn

# pylint: disable=C,R,E1101
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules import Module

from s2cnn.soft.s2_fft import S2_fft_real
from s2cnn.soft.so3_fft import SO3_ifft_real
from s2cnn import s2_mm
from s2cnn import s2_rft


class S2Convolution(Module):
    def __init__(self, nfeature_in, nfeature_out, b_in, b_out, b_inverse, grid, device='cuda:0'):
        '''
        :param nfeature_in: number of input fearures
        :param nfeature_out: number of output features
        :param b_in: input bandwidth (precision of the input SOFT grid)
        :param b_out: output bandwidth
        :param grid: points of the sphere defining the kernel, tuple of (alpha, beta)'s
        '''
        super(S2Convolution, self).__init__()
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.b_in = b_in
        self.b_out = b_out
        self.b_inverse = b_inverse
        self.grid = grid
        self.kernel = Parameter(torch.empty(
            nfeature_in, nfeature_out, len(grid)).uniform_(-1, 1).to(device))
        self.scaling = 1. / \
            math.sqrt(len(self.grid) * self.nfeature_in *
                      (self.b_out ** 4.) / (self.b_in ** 2.))
        self.bias = Parameter(torch.zeros(1, nfeature_out, 1, 1, 1).to(device))

    def forward(self, x):  # pylint: disable=W
        '''
        :x:      [batch, feature_in,  beta, alpha]
        :return: [batch, feature_out, beta, alpha, gamma]
        '''
        if (x.size(1) != self.nfeature_in):
            print(f'ERROR: size mismatch: {x.size(1)} vs. {self.nfeature_in}')
        assert x.size(1) == self.nfeature_in
        assert x.size(2) == 2 * self.b_in
        assert x.size(3) == 2 * self.b_in
        # [l * m, batch, feature_in, complex]
        x = S2_fft_real.apply(x, self.b_out)
        # [l * m, feature_in, feature_out, complex]
        y = s2_rft(self.kernel * self.scaling, self.b_out, self.grid)

        z = s2_mm(x, y)  # [l * m * n, batch, feature_out, complex]
        # [batch, feature_out, beta, alpha, gamma]
        z = SO3_ifft_real.apply(z, self.b_inverse)

        z = z + self.bias

        return z


if __name__ == "__main__":
    b_in = 50
    b_out = 50
    device = torch.device('cpu')
    for_grad = False
    #wigner = _setup_wigner(b_out, nl=b_in, weighted=for_grad, device=device)
