from tqdm.auto import tqdm
from dh_grid import DHGrid
from sub_sampler import SubSampler

import numpy as np

export_ds = '/mnt/data/datasets/nuscenes/processed'
cloud_filename = f"{export_ds}/clouds.npy"
sampled_filename = f"{export_ds}/sampled_clouds.npy"
cloud_features = np.load(cloud_filename)

lf_bw = 100
sampler = SubSampler(cloud_features, lf_bw)
sampled_cloud_features = sampler.compute_output_data()
print("Finished computation.")
print(f"Subsampled shape of the clouds is {sampled_cloud_features.shape}.")
print(f"Exporting them to {sampled_filename}.")

np.save(sampled_filename, sampled_cloud_features)
