import torch
import numpy as np

class Utils:

    @staticmethod
    def fftshift(X):
        '''
        :param X: [l * m * n, batch, feature, complex]
        :return: [l * m * n, batch, feature, complex] centered at floor(l*m*n/2)
        '''
        lmn = X.size(0)
        shift = int(np.floor(lmn / 2))
        return torch.roll(X, dims=0, shifts=shift)

    @staticmethod
    def ifftshift(X):
        '''
        :param X: [l * m * n, batch, feature, complex]
        :return: [l * m * n, batch, feature, complex] reveresed with ceil(l*m*n/2)
        '''
        lmn = X.size(0)
        shift = int(np.ceil(lmn / 2))
        return torch.roll(X, dims=0, shifts=shift)
