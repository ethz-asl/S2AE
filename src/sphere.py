from data_source import DataSource
from dh_grid import DHGrid
import numpy as np
from scipy import spatial
import open3d as o3d
import sys
from tqdm.auto import tqdm
from semantic_classes import SemanticClasses

class Sphere:
    def __init__(self, point_cloud=None, bw=None, features=None, filter=False, normals=False):
        self.ranges = []
        self.intensity = []
        self.normals = []
        self.semantics = []
        self.sampling_grid = []
        if point_cloud is not None:
            if filter:
                point_cloud = self.filter_outliers_from_cloud(point_cloud)
            self.point_cloud = point_cloud
            (self.sphere, self.ranges) = self.__projectPointCloudOnSphere(point_cloud)
            self.intensity = point_cloud[:,3]
            if normals:
                self.normals = self._estimate_normals(point_cloud)
            if point_cloud.shape[1] >= 5:
                self.semantics = point_cloud[:,4]
        elif bw is not None and features is not None:
            self.constructFromFeatures(bw, features)

    def has_semantics(self):
        return len(self.semantics) > 0

    def has_normals(self):
        return len(self.normals) > 0

    def compute_normals(self):
        self.normals = self._estimate_normals(self.point_cloud)

    def filter_outliers_from_cloud(self, pcl, neighbors=30, std=2.0):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl[:, 0:3])
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std)
        return np.take(pcl, ind, axis=0)

    def _estimate_normals(self, pcl):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl[:, 0:3])
        params = o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50)
        pcd.estimate_normals(search_param=params)
#         pcd.estimate_normals()
#         pcd.orient_normals_to_align_with_direction()
        assert pcd.has_normals()

        normals = np.asarray(pcd.normals)
        angle = np.abs(normals[:,0]) + np.abs(normals[:,1])
        return angle

    def constructFromFeatures(self, bw, features):
        self.point_cloud = None
        _, self.sphere = DHGrid.CreateGrid(bw)
        n_grid = 2 * bw
        n_points = n_grid*n_grid

        self.ranges = np.empty([n_points, 1])
        self.intensity = np.empty([n_points, 1])
        self.normals = np.empty([n_points, 1])

        has_semantics = features.shape[0] == 4
        if has_semantics:
            self.semantics = np.empty([n_points, 1])
        cur_idx = 0
        for i in range(n_grid):
            for j in range(n_grid):
                self.ranges[cur_idx] = features[0, i, j]
                self.intensity[cur_idx] = features[1, i, j]
                self.normals[cur_idx] = features[2, i, j]
                if has_semantics:
                    self.semantics[cur_idx] = features[3, i, j]
                cur_idx = cur_idx + 1

    def getProjectedInCartesian(self):
        return self.__convertSphericalToEuclidean(self.sphere, True)

    def sampleUsingGrid(self, grid, invert=True):
        cart_sphere = self.__convertSphericalToEuclidean(self.sphere, invert)
        cart_grid = DHGrid.ConvertGridToEuclidean(grid)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cart_sphere[:, 0:3])
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        kNearestNeighbors = 1
        has_semantics = self.has_semantics()
        has_normals = self.has_normals()
        n_features = 4
        if not has_semantics:
            n_features = n_features - 1
        if not has_normals:
            n_features = n_features - 1
        features = np.ones((n_features, grid.shape[1], grid.shape[2])) * (-1)

        for i in range(grid.shape[1]):
            for j in range(grid.shape[2]):
                [k, nn_idx, nn_dist] = pcd_tree.search_knn_vector_3d(cart_grid[:, i, j], kNearestNeighbors)

                # TODO(lbern): Average over all neighbors
                for k in range(kNearestNeighbors):
                    cur_idx = nn_idx[k]
                    cur_dist = np.absolute(nn_dist[k])
                    if (cur_dist > 0.001):
                        continue

                    range_value = self.ranges[cur_idx]
                    range_value = range_value if not np.isnan(range_value) else -1
                    features[0, i, j] = range_value

                    intensity = self.intensity[cur_idx]
                    intensity = intensity if not np.isnan(intensity) else -1
                    features[1, i, j] = intensity

#                     feature_idx = 2
#                     if has_normals:
#                         normal_angle = self.normals[cur_idx]
#                         normal_angle = normal_angle if not np.isnan(normal_angle) else 0
#                         features[feature_idx, i, j] = normal_angle
#                         feature_idx = feature_idx + 1

#                     if has_semantics:
                    semantics = self.semantics[cur_idx]
                    semantics = semantics if not np.isnan(semantics) else -1
#                         semantics = SemanticClasses.map_sem_kitti_label(semantics) if not np.isnan(semantics) else -1
                    features[2, i, j] = semantics

        return features

    def sampleUsingGrid2(self, grid):
        cart_sphere = self.__convertSphericalToEuclidean(self.sphere, False)
        cart_grid = DHGrid.ConvertGridToEuclidean(grid)

        sys.setrecursionlimit(50000)
        sphere_tree = spatial.cKDTree(cart_sphere[:,0:3])
        p_norm = 2
        n_nearest_neighbors = 1
        features = np.zeros((2, grid.shape[1], grid.shape[2]))
        for i in range(grid.shape[1]):
            for j in range(grid.shape[2]):
                nn_dists, nn_indices = sphere_tree.query(cart_grid[:, i, j], p = p_norm, k = n_nearest_neighbors)
                nn_indices = [nn_indices] if n_nearest_neighbors == 1 else nn_indices

                # TODO(lbern): Average over all neighbors
                for cur_idx in nn_indices:
                    features[0, i, j] = self.ranges[cur_idx]
                    features[1, i, j] = self.intensity[cur_idx]

        return features

    def __projectPointCloudOnSphere(self, cloud):
        # sqrt(x^2+y^2+z^2)
        dist = np.sqrt(cloud[:,0]**2 + cloud[:,1]**2 + cloud[:,2]**2)

        projected = np.empty([len(cloud), 3])
        ranges = np.empty([len(cloud), 1])

        # Some values might be zero or NaN, lets ignore them for now.
        eps = 0.000001
        projected[:,0] = np.arccos(cloud[:,2] / (dist + eps))
        projected[:,1] = np.mod(np.arctan2(cloud[:,1], cloud[:,0]) + 2*np.pi, 2*np.pi)
        ranges[:,0] = dist
        return projected, ranges

    def __convertSphericalToEuclidean(self, spherical, invert):
        cart_sphere = np.zeros([len(spherical), 3])

        # CUSTOM: For some reason the projection need to be inverted here to nicely fit the sph images.
        # Let's see if other datasets require the same modification
        cart_sphere[:,0] = np.multiply(np.sin(spherical[:,0]), np.cos(spherical[:,1]))
        cart_sphere[:,1] = np.multiply(np.sin(spherical[:,0]), np.sin(spherical[:,1]))
        cart_sphere[:,2] = np.cos(spherical[:,0])
        if invert:
            cart_sphere = -cart_sphere
        mask = np.isnan(cart_sphere)
        cart_sphere[mask] = 0
        return cart_sphere

if __name__ == "__main__":
    ds = DataSource("/media/scratch/berlukas/spherical/training")
    ds.load(10)

    sph = Sphere(ds.anchors[0])
    grid = DHGrid.CreateGrid(50)
    features = sph.sampleUsingGrid(grid)
    print("features: ", features.shape)
