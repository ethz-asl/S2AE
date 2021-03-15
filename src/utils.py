import torch

class Utils:
    def roll_n(X, axis, n):
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                      for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                      for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    def fftshift(X):
        # batch*channel*...*2
        real, imag = X.chunk(chunks=2, dim=-1)
        real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)

        for dim in range(2, len(real.size())):
            real = roll_n(real, axis=dim, n=int(np.ceil(real.size(dim) / 2)))
            imag = roll_n(imag, axis=dim, n=int(np.ceil(imag.size(dim) / 2)))

        real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
        X = torch.cat((real,imag),dim=-1)
        return X


    def ifftshift(X):
        # batch*channel*...*2
        real, imag = X.chunk(chunks=2, dim=-1)
        real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)

        for dim in range(len(real.size()) - 1, 1, -1):
            real = roll_n(real, axis=dim, n=int(np.floor(real.size(dim) / 2)))
            imag = roll_n(imag, axis=dim, n=int(np.floor(imag.size(dim) / 2)))

        real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
        X = torch.cat((real, imag), dim=-1)

        return X

    @staticmethod
    def fftshift2(real, imag):
        for dim in range(0, len(real.size())):
            real = torch.roll(real, dims=dim, shifts=real.size(dim)//2)
            imag = torch.roll(imag, dims=dim, shifts=imag.size(dim)//2)
        return torch.cat((real, imag), dim=-1)
