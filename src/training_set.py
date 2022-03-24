import os
from functools import partial

import numpy as np

import open3d as o3d
import pymp
import torch.utils.data
from data_source import DataSource
from dh_grid import DHGrid
from sphere import Sphere
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map

class TrainingSetLidarSeg(torch.utils.data.Dataset):
    def __init__(self, cloud_features, sem_cloud_features):
        self.cloud_features = cloud_features
        self.sem_cloud_features = sem_cloud_features
        assert len(self.cloud_features) == len(self.sem_cloud_features)

    def __getitem__(self, index):
        return self.cloud_features[index], self.sem_cloud_features[index]       

    def __len__(self):
        return len(self.cloud_features)

class TrainingSetFusedSeg(torch.utils.data.Dataset):
    def __init__(self, decoded_features, image_cloud_features, sem_cloud_features):
        self.decoded_features_low = decoded_features
        self.image_cloud_features_high = image_cloud_features
        self.sem_cloud_features_high = sem_cloud_features
        
        self.n_features = len(self.decoded_features_low)
        assert n_features == len(self.image_cloud_features_high)
        assert n_features == len(self.sem_cloud_features_high)

    def __getitem__(self, index):
        return self.decoded_features_low[index], self.image_cloud_features_high[index], self.sem_cloud_features_high[index]

    def __len__(self):
        return self.n_features
    

if __name__ == "__main__":
    cloud_features = [np.zeros((1,2,200,200))]
    sem_cloud_features = [np.zeros((1,2,200,200))]
    ts = TrainingSetLidarSeg(cloud_features, sem_cloud_features)