# pylint: disable=C,R,E1101
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules import Module

from s2cnn.soft.s2_fft import S2_fft_real
from s2cnn.soft.so3_fft import SO3_ifft_real
from s2cnn import s2_mm
from s2cnn import s2_rft

def complex_div(x, y, conj_x=False, conj_y=False):
    '''
    :param x: [i, k, complex] (M, K, 2)
    :param y: [k, j, complex] (K, N, 2)
    :return:  [i, j, complex] (M, N, 2)
    '''
    xr = x[:, :, 0]
    xi = -x[:, :, 1] if conj_x else x[:, :, 1]

    yr = y[:, :, 0]
    yi = -y[:, :, 1] if conj_y else y[:, :, 1]

    nr = torch.mm(yr.transpose(0,1), yr)
    ni = torch.mm(yi.transpose(0,1), yi)
    n = nr + ni

    zr = torch.div(torch.mm(xr, yr) + torch.mm(xi, yi), n)
    zi = torch.div(torch.mm(xi, yr) - torch.mm(xr, yi), n)

    return torch.stack((zr, zi), 2)

def s2_div(x, y, conj_x, conj_y):
    '''
    :param x: [l * m,     batch,      feature_in,  complex]
    :param y: [l * m,     feature_in, feature_out, complex]
    :return:  [l * m * n, batch,      feature_out, complex]
    '''
    from s2cnn.utils.complex import complex_mm

    assert y.size(3) == 2
    assert x.size(3) == 2
    nbatch = x.size(1)
    nfeature_in = x.size(2)
    nfeature_out = y.size(2)
    assert y.size(1) == nfeature_in
    nspec = x.size(0)
    assert y.size(0) == nspec

    if x.is_cuda:
        return foo

    nl = round(nspec**0.5)

    Fz_list = []
    begin = 0
    for l in range(nl):
        L = 2 * l + 1
        size = L

        Fx = x[begin:begin+size]  # [m, batch,      feature_in,  complex]
        Fy = y[begin:begin+size]  # [m, feature_in, feature_out, complex]

        Fx = Fx.view(L * nbatch, nfeature_in, 2)  # [m * batch, feature_in, complex]

        Fy = Fy.transpose(0, 1)  # [feature_in, m, feature_out, complex]
        Fy = Fy.contiguous()
        Fy = Fy.view(nfeature_in, L * nfeature_out, 2)  # [feature_in, m * feature_out, complex]

        Fz = complex_div(Fx, Fy, conj_y=conj_y, conj_x=conj_x)  # [m_x * batch, m_y * feature_out, complex] m_x -> m, m_y -> n
        Fz = Fz.view(L, nbatch, L, nfeature_out, 2)  # [m, batch, n, feature_out, complex]
        Fz = Fz.transpose(1, 2)  # [m, n, batch, feature_out, complex]
        Fz = Fz.contiguous()
        Fz = Fz.view(L * L, nbatch, nfeature_out, 2)  # [m * n, batch, feature_out, complex]

        Fz_list.append(Fz)

        begin += size

    z = torch.cat(Fz_list, 0)  # [l * m * n, batch, feature_out, complex]
    return z


class S2Deconvolution(Module):
    def __init__(self, nfeature_in, nfeature_out, b_in, b_out, grid):
        '''
        :param nfeature_in: number of input fearures
        :param nfeature_out: number of output features
        :param b_in: input bandwidth (precision of the input SOFT grid)
        :param b_out: output bandwidth
        :param grid: points of the sphere defining the kernel, tuple of (alpha, beta)'s
        '''
        super(S2Deconvolution, self).__init__()
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.b_in = b_in
        self.b_out = b_out
        self.grid = grid
        self.kernel = Parameter(torch.empty(nfeature_in, nfeature_out, len(grid)).uniform_(-1, 1))
        self.scaling = 1. / math.sqrt(len(self.grid) * self.nfeature_in * (self.b_out ** 4.) / (self.b_in ** 2.))
        self.bias = Parameter(torch.zeros(1, nfeature_out, 1, 1, 1))

    def forward(self, x):  # pylint: disable=W
        '''
        :x:      [batch, feature_in,  beta, alpha]
        :return: [batch, feature_out, beta, alpha, gamma]
        '''
        assert x.size(1) == self.nfeature_in
        assert x.size(2) == 2 * self.b_in
        assert x.size(3) == 2 * self.b_in
        z = S2_fft_real.apply(x, self.b_out)  # [l * m, batch, feature_in, complex]
        y = s2_rft(self.kernel * self.scaling, self.b_out, self.grid)  # [l * m, feature_in, feature_out, complex]

        #zy = s2_mm_inv(z, y, conj_x=False, conj_y=False)  # [l * m * n, batch, feature_out, complex]
        #yy = s2_mm_inv(y, y, conj_x=True, conj_y=False)  # [l * m * n, batch, feature_out, complex]
        z = s2_div(z, y, conj_y = True)

        z = SO3_ifft_real.apply(z)  # [batch, feature_out, beta, alpha, gamma]

        z = z + self.bias

        return z

if __name__ == "__main__":
    a = torch.zeros([1, 1, 2]) # batch, feature, complex
    b = torch.zeros([1, 1, 2]) # batch, feature, complex
    c = torch.zeros([1, 1, 2]) # batch, feature, complex
    # a = 3+5j
    a[0,0,0] = 3
    a[0,0,1] = 5
    # b = 2+7j
    b[0,0,0] = 2
    b[0,0,1] = 7
    # c = 5 + 0j
    c[0,0,0] = 5

    print(f"Calculting a/b with a {a} and b {b}.")
    z = complex_div(a, b)
    print(f"The result z is {z}.")

    print(f"Calculting a/c with a {a} and c {c}.")
    z = complex_div(a, c)
    print(f"The result z is {z}.")
