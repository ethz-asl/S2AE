# pylint: disable=C,R,E1101
import math
import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
import matplotlib.pyplot as plt

from s2cnn.soft.so3_fft import SO3_fft_real, SO3_ifft_real, so3_rfft
from s2cnn.soft.s2_fft import S2_fft_real,s2_fft
from s2cnn import s2_rft, so3_rft

from utils import Utils

class SO3_spectral_pool_symmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b_out=None):  # pylint: disable=W
        '''
        :param ctx: context
        :param x: input data [batch, feature_in, beta, alpha, gamma]
        '''
        ctx.b_out = b_out
        ctx.b_in = x.size(-1) // 2
        ctx.f_in = x.size(1)

        # Transform the input to the spherical domain.
        # Also, shift the DC component to the center.
        x = so3_rfft(x, b_out=ctx.b_in)
        X, center = Utils.fftshift(x)

        # Low-pass filter the signal.
        lhs,rhs = Utils.compute_bounds_SO3(b_out)
        ctx.lb = int(center - lhs)
        ctx.ub = int(center + rhs)
        X = X[ctx.lb:ctx.ub, :, :, :]

        # Shift the signals back and perform a inverse transform.
        X, _ = Utils.ifftshift(X)
        ctx.device = x.device
        return SO3_ifft_real.apply(X)  # [batch, feature_out, beta, alpha, gamma]

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=W
        X = so3_rfft(grad_output, b_out=ctx.b_out)
        X, _ = Utils.fftshift(X)

        # Zero-pad the signal to the bigger size.
        samples = Utils.compute_samples_SO3(ctx.b_in)
        Y = torch.zeros((samples, X.size(1), X.size(2), X.size(3)), device=ctx.device)
        center = samples // 2
        lhs,rhs = Utils.compute_bounds_SO3(ctx.b_out)
        ctx.lb = int(center - lhs)
        ctx.ub = int(center + rhs)
        Y[ctx.lb:ctx.ub, :, :, :] = X[:,:,:,:]

        # Shift the signals back and perform a inverse transform.
        X, _ = Utils.ifftshift(Y)
        return SO3_ifft_real.apply(X), None  # [batch, feature_out, beta, alpha, gamma]

class SO3_spectral_pool_left(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b_out=None):  # pylint: disable=W
        '''
        :param ctx: context
        :param x: input data [batch, feature_in, beta, alpha, gamma]
        '''
        ctx.b_out = b_out
        ctx.b_in = x.size(-1) // 2
        ctx.f_in = x.size(1)

        # Transform the input to the spherical domain.
        X = so3_rfft(x, b_out=ctx.b_in)

        # Low-pass filter the signal.
        lhs,rhs = Utils.compute_bounds_SO3_left_full(b_out)
        ctx.lb = lhs
        ctx.ub = rhs
        X = X[ctx.lb:ctx.ub, :, :, :]

        ctx.device = x.device

        # Perform a inverse transform.
        return SO3_ifft_real.apply(X)  # [batch, feature_out, beta, alpha, gamma]

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=W
        X = so3_rfft(grad_output, b_out=ctx.b_out)

        # Zero-pad the signal to the bigger size.
        samples = Utils.compute_samples_SO3(ctx.b_in)
        Y = torch.zeros((samples, X.size(1), X.size(2), X.size(3)), device=ctx.device)
        Y[ctx.lb:ctx.ub, :, :, :] = X[:,:,:,:]

        # Perform an inverse transform.
        return SO3_ifft_real.apply(Y), None  # [batch, feature_out, beta, alpha, gamma]

class SO3Pooling(Module):
    def __init__(self, b_in, b_out):
        '''
        :param b_in: input bandwidth (precision of the input SOFT grid)
        :param b_out: output bandwidth
        '''
        super(SO3Pooling, self).__init__()
        self.b_in = b_in
        self.b_out = b_out
        assert self.b_out < self.b_in

    def forward(self, x):  # pylint: disable=W
        '''
        :x:      [batch, feature_in,  beta, alpha, gamma]
        :return: [batch, feature_out, beta, alpha, gamma]
        '''
        assert x.size(2) == 2 * self.b_in
        assert x.size(3) == 2 * self.b_in
        assert x.size(4) == 2 * self.b_in
        return SO3_spectral_pool_symmetric.apply(x, self.b_out)

def plot_signal(X):
    F0 = X[:,0,0,:]
    F1 = X[:,0,1,:]
    X_e0 = torch.view_as_complex(F0).abs()
    X_e1 = torch.view_as_complex(F1).abs()
    plt.plot(X_e0)
    plt.plot(X_e1)
    plt.show()

def forward_test(x_init, b_in, b_out):
    x = SO3_fft_real.apply(x_init, b_in)
    #x = so3_rfft(x_init, b_out=b_in)
    x[0,:,:,:] = 0.008
    print(f"[forward] x shape after transform is {x.size()}") # [l*m*n, batch, feature_in, complex]

    X, center = Utils.fftshift(x)

    # Low-pass filter the signal.
    print(f"[forward] X shape before LP is {X.size()}")
    lhs,rhs = Utils.compute_bounds_SO3(b_out)
    lb = int(center - lhs)
    ub = int(center + rhs)
    X = X[lb:ub, :, :, :]
    #X[0:lb, :, :, :] = 0
    #X[ub:, :, :, :] = 0
    print(f"[forward] LHS: {lhs}, RHS: {rhs}, lb = {lb}, ub = {ub}, center = {center} and X is {X.size()}")

    X, _ = Utils.ifftshift(X)
    return SO3_ifft_real.apply(X)  # [batch, feature_out, beta, alpha, gamma]

def backward_test(x_fwd, b_in, b_out):
    print(f'[backward] input x shape is {x_fwd.shape}')
    X = SO3_fft_real.apply(x_fwd, b_out)
    X, center = Utils.fftshift(X)

    samples = Utils.compute_samples_SO3(b_in)
    lhs, rhs = Utils.compute_bounds_SO3(b_out)
    center = samples // 2
    lb = int(center - lhs)
    ub = int(center + rhs)
    print(f"[backward] LHS: {lhs}, RHS: {rhs}, lb = {lb}, ub = {ub}, center = {center} and X is {X.size()}")

    Y = torch.zeros((samples, X.size(1), X.size(2), X.size(3))) # [l*m*n, batch, feature_in, complex]
    Y[lb:ub, :, :, :] = X[:,:,:,:]
    X, _ = Utils.ifftshift(Y)
    return SO3_ifft_real.apply(X)  # [batch, feature_out, beta, alpha, gamma]

if __name__ == "__main__":
    b_in = 10
    b_out = b_in - 5
    x_init = torch.rand(1, 2, 2*b_in, 2*b_in, 2*b_in)   # [batch, feature_in, beta, alpha, gamma]

    x_fwd = forward_test(x_init, b_in, b_out)
    print(f"x forward shape: {x_fwd.size()} x_init = {x_init.size()}")

    print('---------------------------------------------------------')

    x_bwd = backward_test(x_fwd, b_in, b_out)
    print(f"x backward shape: {x_bwd.size()} x_init = {x_init.size()}")
