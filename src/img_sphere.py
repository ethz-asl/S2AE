from data_source import DataSource
from dh_grid import DHGrid
import numpy as np
from scipy import spatial
import open3d as o3d
import sys
from tqdm.auto import tqdm

class ImageSphere:
    def __init__(self, sph_image, bw=100):
        self.sph_image = sph_image
        self.intensity = sph_image[:,3]

    def sampleUsingGrid(self, grid):
        cart_sphere = self.sph_image
        cart_grid = DHGrid.ConvertGridToEuclidean(grid)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cart_sphere[:, 0:3])
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        kNearestNeighbors = 1
        features = np.zeros((1, grid.shape[1], grid.shape[2]))
        for i in range(grid.shape[1]):
            for j in range(grid.shape[2]):
                [k, nn_idx, _] = pcd_tree.search_knn_vector_3d(cart_grid[:, i, j], kNearestNeighbors)

                # TODO(lbern): Average over all neighbors
                for cur_idx in nn_idx:
                    intensity = self.intensity[cur_idx]
                    intensity = intensity if not np.isnan(intensity) else 0
                    features[0, i, j] = intensity

        return features


if __name__ == "__main__":
    print("Not yet implemented")
