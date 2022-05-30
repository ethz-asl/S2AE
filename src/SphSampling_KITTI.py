#!/usr/bin/env python
# coding: utf-8

# ## Create Sampled Dataset of KITTI
# 

# In[6]:


import argparse
import os
import yaml
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map
from matplotlib import cm
from functools import partial
import concurrent.futures

from sphere import Sphere
from dh_grid import DHGrid
from laserscan import SemLaserScan
from utils import Utils


def load_sequence(dataroot, sequence):
    scan_paths = f'{dataroot}/{sequence}/velodyne'
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()
    
    label_paths = f'{dataroot}/{sequence}/labels'
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn]
    label_names.sort()    
    assert len(label_names) == len(scan_names)
    print(f'Found {len(scan_names)} pointclouds and labels for sequence {sequence}.')
    return scan_names, label_names

def load_config_file(config_file):    
    try:        
        CFG = yaml.safe_load(open(config_file, 'r'))        
        return CFG
    except Exception as e:
        print(e)        
        return None
    
def parse_calibration(dataroot, seq):
    filename = dataroot + '/' + seq + '/calib.txt'
    calib = {}
    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose
    calib_file.close()
    return calib

# dataroot = '/mnt/data/datasets/KITTI/sequences'
# dataroot = '/media/berlukas/Data/data/datasets/KITTI/sequences'
# dataroot = '/media/berlukas/SSD_1TB/data_odometry_velodyne/dataset/sequences'
dataroot = '/media/scratch/berlukas/nuscenes/kitti'

# sequences = ['01', '02', '03']
sequences = ['08']
config_file = '../config/semantic-kitti.yaml'
# export_ds = '/media/berlukas/Data/data/nuscenes'
export_ds = '/media/scratch/berlukas/nuscenes/kitti_processed'

print(f'Setting dataroot to {dataroot}.')
print(f'Setting export path to {export_ds}.')
print(f'Configured {len(sequences)} sequences.')
print(f'Configured config file {config_file}')


# In[10]:


all_sem_clouds = []

def progresser(sample_idx, grid, auto_position=True, write_safe=False, blocking=True, progress=False):    
#     sample_sphere = Sphere(sample)
    sample_sphere = all_sem_clouds[sample_idx]
    features = sample_sphere.sampleUsingGrid(grid)
    return features

def parse_poses(dataroot, seq, calib):
    file = dataroot + '/' + seq + '/poses.txt'
    poses_arr = pd.read_csv(file, delimiter=' ', comment='#', header=None).to_numpy()    
    poses = [np.array([[r[0], r[1], r[2], r[3]],
                   [r[4], r[5], r[6], r[7]],
                   [r[8], r[9], r[10], r[11]],
                   [0, 0, 0, 1]]) for r in poses_arr]
    
    T_C_L = calib['Tr']
    n_poses = len(poses)
    for i in range(0, n_poses):    
        T_G_C = poses[i]
        poses[i] = T_G_C @ T_C_L
    return poses

def get_pointcloud_at(scan, name, label):
    pointclouds = []            
    scan.open_scan(name)
    scan.open_label(label)
    scan.colorize()

    pc = np.column_stack((scan.points, scan.remissions, scan.sem_label))
    mask = pc[:,4] > 0 # Filter based on labeled data.        
    pointclouds.append(pc[mask])
    return pointclouds

def get_map_at(scan, names, labels, indices):
    pointclouds = []    
    for idx in indices:
        scan.open_scan(names[idx])
        scan.open_label(labels[idx])
        scan.colorize()
        
        pc = np.column_stack((scan.points, scan.remissions, scan.sem_label))
        mask = pc[:,4] > 0 # Filter based on labeled data.        
        pointclouds.append(pc[mask])
    return pointclouds

def retrieve_poses_at(all_poses, indices):
    poses = []
    for idx in indices:
        poses.append(all_poses[idx])
    return poses

def combine_pointclouds(pointclouds, poses):
    n_data = len(poses)
    
    pivot = n_data // 2  
    T_G_L_pivot = poses[pivot]
    T_L_pivot_G = np.linalg.inv(T_G_L_pivot)

    acc_points = pointclouds[pivot]
    for i in range(0, n_data):
        if i == pivot:
            continue

        T_G_L = poses[i]
        T_L_pivot_L = T_L_pivot_G @ T_G_L

        points = Utils.transform_pointcloud(pointclouds[i], T_L_pivot_L)
        acc_points = np.append(acc_points, points, axis=0)                    
    
    return acc_points

CFG = load_config_file(config_file)
color_dict = CFG["color_map"]
nclasses = len(color_dict)
scan = SemLaserScan(nclasses, color_dict, project=False)
bw = 120
assert CFG is not None
  
grid, _ = DHGrid.CreateGrid(bw)
for seq in sequences:
    print(f'Loading sequence {seq}.')    
    scan_names, label_names = load_sequence(dataroot, seq)
    calib = parse_calibration(dataroot, seq)
    poses = parse_poses(dataroot, seq, calib)    
    n_scans = len(scan_names)    
    print(f'This sequence has {len(poses)} data elements.')
        
    all_sem_clouds = []
    n_scans = 5
    for i in range(0, n_scans):                
        pointcloud = get_pointcloud_at(scan, scan_names[i], label_names[i])        
        all_sem_clouds.append(Sphere(pointcloud[0]))
        
    print(f"Loading complete. Computing features...")
    # parallel
    sem_idx = np.arange(0, len(all_sem_clouds))
    sample_func = partial(progresser, grid=grid)
    sem_features = process_map(sample_func, sem_idx, max_workers=16)            

    filename = f"{export_ds}/clouds-{seq}.npy"
    np.save(filename, sem_features)
    print(f"Wrote features to {filename}.")
