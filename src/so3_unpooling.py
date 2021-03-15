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


class SO3_spectral_unpool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b_out=None):  # pylint: disable=W
        '''
        :param ctx: context
        :param x: input data [batch, feature_in, beta, alpha, gamma]
        '''
        ctx.b_out = b_out
        ctx.b_in = x.size(-1) // 2

        # Transform the input to the spherical domain.
        # Also, shift the DC component to the center.
        x = so3_rfft(x, b_out=ctx.b_in)
        X, _ = Utils.fftshift(x)

        # Zero-pad the signal to the bigger size.
        samples = Utils.compute_samples_SO3(b_out)
        center = samples // 2
        Y = torch.zeros((samples, X.size(1), X.size(2), X.size(3)))
        lhs,rhs = Utils.compute_bounds_SO3(b_in)

        ctx.lb = int(center - lhs)
        ctx.ub = int(center + rhs)
        Y[ctx.lb:ctx.ub, :, :, :] = X[:,:,:,:]

        # Shift the signals back and perform a inverse transform.
        X, _ = Utils.ifftshift(Y)
        return SO3_ifft_real.apply(X)  # [batch, feature_out, beta, alpha, gamma]

class SO3Unpooling(Module):
    def __init__(self, nfeature_in, nfeature_out, b_in, b_out):
        '''
        :param nfeature_in: number of input fearures
        :param nfeature_out: number of output features
        :param b_in: input bandwidth (precision of the input SOFT grid)
        :param b_out: output bandwidth
        :param grid: points of the SO(3) group defining the kernel, tuple of (alpha, beta, gamma)'s
        '''
        super(SO3Unpooling, self).__init__()
        self.nfeature_in = nfeature_in
        self.b_in = b_in
        self.b_out = b_out
        assert self.b_out > self.b_in

    def forward(self, x):  # pylint: disable=W
        '''
        :x:      [batch, feature_in,  beta, alpha, gamma]
        :return: [batch, feature_out, beta, alpha, gamma]
        '''
        assert x.size(1) == self.nfeature_in
        assert x.size(2) == 2 * self.b_in
        assert x.size(3) == 2 * self.b_in
        assert x.size(4) == 2 * self.b_in
        return SO3_spectral_unpool.apply(x, self.b_out)

if __name__ == "__main__":
    b_in = 30
    x_init = torch.rand(1, 2, 60, 60, 60)  # [batch, feature_in, beta, alpha, gamma]
    x = SO3_fft_real.apply(x_init, b_in)
    print(f"x shape after transform is {x.size()}") # [l*m*n, batch, feature_in, complex]

    X, center = Utils.fftshift(x)
    F0 = X[:,0,0,:]
    F1 = X[:,0,1,:]
    X_e0 = torch.view_as_complex(F0).abs()
    X_e1 = torch.view_as_complex(F1).abs()
    print(f"feature X shape: {X_e0.size()}")
    plt.plot(X_e0)
    plt.plot(X_e1)
    #plt.show()

    # Low-pass filter the signal.
    print(f"X shape before LP is {X.size()}")
    lhs,rhs = Utils.compute_bounds_SO3(b_in - 5)
    lb = int(center - lhs)
    ub = int(center + rhs)
    #X = X[lb:ub, :, :, :]
    X[0:lb, :, :, :] = 0
    X[lb:, :, :, :] = 0
    print(f"LHS: {lhs}, RHS: {rhs}, lb = {lb}, ub = {ub}, center = {center} and X is {X.size()}")

    X, _ = Utils.ifftshift(X)
    print(f"shifted signal {X.size()}")
    F0 = X[:,0,0,:]
    F1 = X[:,0,1,:]
    X_e0 = torch.view_as_complex(F0).abs()
    X_e1 = torch.view_as_complex(F1).abs()
    print(f"feature X shape: {X_e0.size()}")
    plt.plot(X_e0)
    plt.plot(X_e1)
    #plt.show()

    z = SO3_ifft_real.apply(X)  # [batch, feature_out, beta, alpha, gamma]
    print(f"Reverse transformed features shape: {z.size()} x_init = {x_init.size()}")
