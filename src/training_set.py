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
    def __init__(self, bw, cloud_features, sem_cloud_features):
        self.bw = bw
        self.cloud_features = cloud_features
        self.sem_cloud_features = sem_cloud_features
        assert len(self.cloud_features) == len(self.sem_cloud_features)

    def __getitem__(self, index):
        return self.cloud_features[index], self.sem_cloud_features[index]       

    def __len__(self):
        return len(self.cloud_features)

    

if __name__ == "__main__":
    bw = 100
    cloud_features = [np.zeros((1,2,200,200))]
    sem_cloud_features = [np.zeros((1,2,200,200))]
    ts = TrainingSetLidarSeg(bw, cloud_features, sem_cloud_features)