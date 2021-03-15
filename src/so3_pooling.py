# pylint: disable=C,R,E1101
import math
import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
import matplotlib.pyplot as plt

from s2cnn.soft.so3_fft import SO3_fft_real, SO3_ifft_real
from s2cnn.soft.s2_fft import S2_fft_real,s2_fft
from s2cnn import so3_rft
from s2cnn import s2_rft

from utils import Utils


class SO3_spectral_pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b_out=None):  # pylint: disable=W
        '''
        :param ctx: context
        :param x: input data [batch, feature_in, beta, alpha, gamma]
        '''
        ctx.b_out = b_out
        ctx.b_in = x.size(-1) // 2
        ctx.low_pass_b_out = ctx.b_out // 2

        x = so3_rfft(x, b_out=ctx.b_out)

        # shift x to center?
        x = torch.view_as_complex(x)
        #x = tfft.fftshift(x)
        #plt.plot(x[:,1,1])



        #return

    @staticmethod
    def backward(self, grad_output):  # pylint: disable=W
        # ifft of grad_output is not necessarily real, therefore we cannot use rifft
        return so3_ifft(grad_output, for_grad=True, b_out=self.b_in)[..., 0], None

class SO3Pooling(Module):
    def __init__(self, nfeature_in, nfeature_out, b_in, b_out, grid):
        '''
        :param nfeature_in: number of input fearures
        :param nfeature_out: number of output features
        :param b_in: input bandwidth (precision of the input SOFT grid)
        :param b_out: output bandwidth
        :param grid: points of the SO(3) group defining the kernel, tuple of (alpha, beta, gamma)'s
        '''
        super(SO3Pooling, self).__init__()
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.b_in = b_in
        self.b_out = b_out

    def forward(self, x):  # pylint: disable=W
        '''
        :x:      [batch, feature_in,  beta, alpha, gamma]
        :return: [batch, feature_out, beta, alpha, gamma]
        '''
        assert x.size(1) == self.nfeature_in
        assert x.size(2) == 2 * self.b_in
        assert x.size(3) == 2 * self.b_in
        assert x.size(4) == 2 * self.b_in

        x = SO3_fft_real.apply(x, self.b_out)  # [l * m * n, batch, feature_in, complex]
        y = so3_rft(self.kernel * self.scaling, self.b_out, self.grid)  # [l * m * n, feature_in, feature_out, complex]
        assert x.size(0) == y.size(0)
        assert x.size(2) == y.size(1)
        z = so3_mm(x, y)  # [l * m * n, batch, feature_out, complex]
        assert z.size(0) == x.size(0)
        assert z.size(1) == x.size(1)
        assert z.size(2) == y.size(2)
        z = SO3_ifft_real.apply(z)  # [batch, feature_out, beta, alpha, gamma]

        return z

if __name__ == "__main__":
    x = torch.rand(1, 2, 12, 12, 12)  # [batch, feature_in, beta, alpha, gamma]
    x = SO3_fft_real.apply(x, 30)
    print(f"x shape after transform is {x.size()}") # [l*m*n, batch, feature_in, complex]

    #X = torch.view_as_complex(x[:, 0, 0,:])
    X = x.view(x.size(1), x.size(2), x.size(0), 2) # [batch, feature_in, l*m*n, complex]
    real = X[0,0,:,0]
    imag = X[0,0,:,1]

    lmn = X.size(-2)
    shift = lmn // 2 if lmn % 2 == 0 else (lmn + 1) // 2
    print(f"real and imag shape are {real.size()} and {imag.size()}")
    print(f"")

    real = torch.roll(real, dims=0, shifts=shift)
    imag = torch.roll(imag, dims=0, shifts=shift)
    X = torch.stack((real, imag), dim=-1)
    print(f"size of shifted is {X.size()}")

    #X = Utils.fftshift(X)
    #X = Utils.fftshift2(real, imag)
    #X = torch.cat((real, imag), dim=-1)
    #F =
    X_e = torch.view_as_complex(X).abs() ** 2
    print(f"feature X shape: {X_e.size()}")
    plt.plot(X_e)
    plt.show()


    #x = torch.rand(1, 2, 12, 12, 2)  # [..., beta, alpha, complex]
    #z1 = s2_fft(x, b_out=5)
