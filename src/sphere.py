from data_source import DataSource
from dh_grid import DHGrid
import numpy as np
from scipy import spatial
import open3d as o3d
import sys
from tqdm.auto import tqdm


class Sphere:
    def __init__(self, point_cloud=None, bw=None, features=None, normalize=False):
        if point_cloud is not None:
            if normalize:
                xyz = point_cloud[:, 0:3]
                mu = xyz.mean()
                point_cloud[:, 0:3] = (xyz - mu)

            self.point_cloud = point_cloud

            (self.sphere, self.ranges) = self.__projectPointCloudOnSphere(point_cloud)
            self.intensity = point_cloud[:, 3]
            self.semantics = []
            if point_cloud.shape[1] >= 5:
                self.semantics = point_cloud[:, 4]
        elif bw is not None and features is not None:
            self.constructFromFeatures(bw, features)

    def constructFromFeatures(self, bw, features):
        self.point_cloud = None
        _, self.sphere = DHGrid.CreateGrid(bw)
        n_grid = 2 * bw
        n_points = n_grid*n_grid

        self.ranges = np.empty([n_points, 1])
        self.intensity = np.empty([n_points, 1])
        cur_idx = 0
        for i in range(n_grid):
            for j in range(n_grid):
                self.ranges[cur_idx] = features[0, i, j]
                self.intensity[cur_idx] = features[1, i, j]
                cur_idx = cur_idx + 1

    def getProjectedInCartesian(self):
        return self.__convertSphericalToEuclidean(self.sphere)

    def has_semantics(self):
        return len(self.semantics) > 0

    def sampleUsingGrid(self, grid, invert=True):
        cart_sphere = self.__convertSphericalToEuclidean(self.sphere, invert)
        cart_grid = DHGrid.ConvertGridToEuclidean(grid)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cart_sphere[:, 0:3])
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        kNearestNeighbors = 1
        has_semantics = self.has_semantics()
        n_features = 2
        if has_semantics:
            n_features = n_features + 1
        features = np.ones((n_features, grid.shape[1], grid.shape[2])) * (-1)

        for i in range(grid.shape[1]):
            for j in range(grid.shape[2]):
                [k, nn_idx, nn_dist] = pcd_tree.search_knn_vector_3d(
                    cart_grid[:, i, j], kNearestNeighbors)

                for k in range(kNearestNeighbors):
                    cur_idx = nn_idx[k]
                    cur_dist = np.absolute(nn_dist[k])
                    if (cur_dist > 0.001):
                        continue

                    range_value = self.ranges[cur_idx]
                    range_value = range_value if not np.isnan(
                        range_value) else -1
                    features[0, i, j] = range_value

                    intensity = self.intensity[cur_idx]
                    intensity = intensity if not np.isnan(intensity) else -1
                    features[1, i, j] = intensity

                    if has_semantics:
                        semantics = self.semantics[cur_idx]
                        semantics = semantics if not np.isnan(
                            semantics) else -1
                        features[2, i, j] = semantics

        return features

    def __projectPointCloudOnSphere(self, cloud):
        # sqrt(x^2+y^2+z^2)
        dist = np.sqrt(cloud[:, 0]**2 + cloud[:, 1]**2 + cloud[:, 2]**2)
        #dist = np.sqrt(np.power(sph_image_cart[:,0],2) + np.power(sph_image_cart[:,1],2) + np.power(sph_image_cart[:,2],2))

        projected = np.empty([len(cloud), 3])
        ranges = np.empty([len(cloud), 1])

        # Some values might be zero or NaN, lets ignore them for now.
        with np.errstate(divide='ignore', invalid='ignore'):
            projected[:, 0] = np.arccos(cloud[:, 2] / dist)
            projected[:, 1] = np.mod(np.arctan2(
                cloud[:, 1], cloud[:, 0]) + 2*np.pi, 2*np.pi)
            ranges[:, 0] = dist
        return projected, ranges

    def __convertSphericalToEuclidean(self, spherical, invert=False):
        cart_sphere = np.zeros([len(spherical), 3])

        # CUSTOM: For some reason the projection need to be inverted here to nicely fit the sph images.
        # Let's see if other datasets require the same modification
        cart_sphere[:, 0] = np.multiply(
            np.sin(spherical[:, 0]), np.cos(spherical[:, 1]))
        cart_sphere[:, 1] = np.multiply(
            np.sin(spherical[:, 0]), np.sin(spherical[:, 1]))
        cart_sphere[:, 2] = np.cos(spherical[:, 0])
        if invert:
            cart_sphere = -cart_sphere
        mask = np.isnan(cart_sphere)
        cart_sphere[mask] = 0
        return cart_sphere

    def __convertEuclideanToSpherical(self, euclidean):
        sphere = np.zeros([len(euclidean), 2])
        dist = np.sqrt(np.power(sph_image_cart[:, 1], 2) + np.power(
            sph_image_cart[:, 2], 2) + np.power(sph_image_cart[:, 3], 2))
        sphere[:, 0] = np.arccos()


if __name__ == "__main__":
    ds = DataSource("/media/scratch/berlukas/spherical/training")
    ds.load(10)

    sph = Sphere(ds.anchors[0])
    grid = DHGrid.CreateGrid(50)
    features = sph.sampleUsingGrid2(grid)
    print("features: ", features.shape)
