import torch
import numpy as np
import open3d as o3d

class Utils:

    @staticmethod
    def fftshift(X):
        '''
        :param X: [l * m * n, batch, feature, complex]
        :return: [l * m * n, batch, feature, complex] centered at floor(l*m*n/2)
        '''
        lmn = X.size(0)
        shift = int(np.floor(lmn / 2))
        return torch.roll(X, dims=0, shifts=shift), shift

    @staticmethod
    def ifftshift(X):
        '''
        :param X: [l * m * n, batch, feature, complex]
        :return: [l * m * n, batch, feature, complex] reveresed with ceil(l*m*n/2)
        '''
        lmn = X.size(0)
        shift = int(np.ceil(lmn / 2))
        return torch.roll(X, dims=0, shifts=shift), shift

    @staticmethod
    def compute_samples_SO3(bw):
        '''
        Computes the number of samples for a given BW.
        '''
        return bw * (4 * bw ** 2 - 1) // 3

    @staticmethod
    def compute_bounds_SO3(bw):
        '''
        Computes the lower and upper bounds such that the Bw fits between them.
        '''
        samples = Utils.compute_samples_SO3(bw)
        half_samples = samples / 2
        lower_bound = int(np.floor(half_samples))
        upper_bound = int(np.ceil(half_samples))

        return lower_bound, upper_bound
    
    @staticmethod
    def compute_bounds_SO3_left_full(bw):
        '''
        Computes the lower and upper bounds such that the Bw fits between them.
        '''
        samples = Utils.compute_samples_SO3(bw)
        lower_bound = 0
        upper_bound = int(samples)

        return lower_bound, upper_bound
    
    @staticmethod
    def transform_pointcloud(cloud, T):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[:, 0:3])
        pcd.transform(T)
        dst = np.asarray(pcd.points)
        return np.column_stack((dst, cloud[:, 3], cloud[:, 4], cloud[:, 5]))

if __name__ == "__main__":
    bw = 30
    gt_samples = 35990
    samples = Utils.compute_samples_SO3(bw)
    assert samples == gt_samples
    lb,ub = Utils.compute_bounds_SO3(bw)
    assert lb == ub == gt_samples / 2

    bw = 25
    gt_samples = 20825
    samples = Utils.compute_samples_SO3(bw)
    assert samples == 20825
    lb,ub = Utils.compute_bounds_SO3(bw)
    assert (lb + ub) == gt_samples
    assert lb < ub
