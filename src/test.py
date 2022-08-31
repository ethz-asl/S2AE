#!/usr/bin/env python
# coding: utf-8

# # Test Existing Network

# In[1]:


import math
import sys
import time
import os

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm.auto import tqdm

from data_splitter import DataSplitter
from training_set import TrainingSetLidarSeg
from external_splitter import ExternalSplitter
from loss import *
from model import Model
from metrics import *
from average_meter import AverageMeter
from iou import IoU

print(f"Initializing CUDA...")
#torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

print(f"Setting parameters...")
bandwidth = 50
batch_size = 10

num_workers = 32
n_classes = 6
device_ids = [0,1]

print(f"Initializing data structures...")
criterion = WceLovasz()

writer = SummaryWriter()

model = Model(bandwidth=bandwidth, n_classes=n_classes).cuda(0)
net = nn.DataParallel(model, device_ids = device_ids).to(0)

# chkp = './checkpoints/euler_lidarseg_20220527192933_13.pth'
chkp = './checkpoints/test_lidarseg_20220824073410_12.pth'


print(f'Loading checkpoint from {chkp}...')
checkpoint = torch.load(chkp)

print('Loading trained model weights...')
net.load_state_dict(checkpoint['model_state_dict'])

print(f"All instances initialized.")


# export_ds = '/mnt/data/datasets/nuscenes/processed'
# export_ds = '/media/scratch/berlukas/nuscenes'
export_ds = '/cluster/work/riner/users/berlukas'

data_ds = f'{export_ds}/val_kitti'
data_samples = os.listdir(data_ds)
data_sets = []
for sample in data_samples:
    sem_clouds_filename = f'{data_ds}/{sample}'
    print(f'Loading from sem clouds from {sem_clouds_filename}')
    data_features = np.load(sem_clouds_filename)

    sem_data_features = np.copy(data_features[:, 1, :, :])
    data_features = data_features[:, 0, :, :]
    data_features = np.reshape(data_features, (-1, 1, 2*bandwidth, 2*bandwidth))
    print(f"Shape clouds is {data_features.shape} and sem clouds is {sem_data_features.shape}")
    data_set = TrainingSetLidarSeg(data_features, sem_data_features)
    data_sets.append(data_set)

n_data_sets = len(data_sets)
assert n_data_sets > 0
print(f'Loaded {n_data_sets} test sets. Setting up the test loaders now.')
print('\n')

data_loaders = [data_loader]
for data_i in range(1, n_data_sets):
    split = ExternalSplitter(None, data_sets[data_i])
    _, data_loader = split.get_split(batch_size=batch_size, num_workers=num_workers)
    
    data_size = data_size + split.get_val_size()
    data_loaders.append(data_loader)


print("Dataset size for testing: ", data_size)

def test_lidarseg(net, criterion, writer):
    all_decoded_clouds = [None] * data_size
    all_gt_clouds = [None] * data_size
    k = 0
    avg_pixel_acc = AverageMeter()
    avg_pixel_acc_per_class = AverageMeter()
    avg_jacc = AverageMeter()
    avg_dice = AverageMeter()
    n_iter = 0

    ignore_index = 0
    metric = IoU(n_classes, ignore_index=ignore_index)

    net.eval()
    with torch.no_grad():            
        for data_loader in data_loaders:
            for batch_idx, (cloud, lidarseg_gt) in enumerate(data_loader):
                cloud, lidarseg_gt = cloud.cuda().float(), lidarseg_gt.cuda().long()
                enc_dec_cloud = net(cloud)
                
                pred_segmentation = torch.argmax(enc_dec_cloud, dim=1)
                mask = lidarseg_gt <= 0
                pred_segmentation[mask] = 0
                lidarseg_gt[mask] = 0

                pixel_acc, pixel_acc_per_class, jacc, dice = eval_metrics(lidarseg_gt, pred_segmentation, num_classes = n_classes)
                avg_pixel_acc.update(pixel_acc)
                avg_pixel_acc_per_class.update(pixel_acc_per_class)
                avg_jacc.update(jacc)
                avg_dice.update(dice)
                
                writer.add_scalar('Test/PixelAccuracy', pixel_acc, n_iter)   
                writer.add_scalar('Test/PixelAccuracyPerClass', pixel_acc_per_class, n_iter)   
                writer.add_scalar('Test/JaccardIndex', jacc, n_iter)
                writer.add_scalar('Test/DiceCoefficient', dice, n_iter)  

                
                n_batch = enc_dec_cloud.shape[0]
                for i in range(0, n_batch):                
                    all_decoded_clouds[k] = pred_segmentation.cpu().data.numpy()[i,:,:]
                    all_gt_clouds[k] = lidarseg_gt.cpu().data.numpy()[i,:,:]
                    k = k + 1     

                    pred = torch.reshape(pred_segmentation[i, :, :], [1, 2*bandwidth, 2*bandwidth]).int()
                    gt = torch.reshape(lidarseg_gt[i, :, :], [1, 2*bandwidth, 2*bandwidth]).int()
                    metric.add(pred, gt)
                n_iter += 1
                
            writer.add_scalar('Test/AvgPixelAccuracy', avg_pixel_acc.avg, n_iter)   
            writer.add_scalar('Test/AvgPixelAccuracyPerClass', avg_pixel_acc_per_class.avg, n_iter)   
            writer.add_scalar('Test/AvgJaccardIndex', avg_jacc.avg, n_iter)
            writer.add_scalar('Test/AvgDiceCoefficient', avg_dice.avg, n_iter)  

            print('============================================================================')
            print(f'Validation Results for Validation Loader: {data_samples[k]}.')

            print('\n')
            print(f'[Testing] Average Pixel Accuracy: {avg_pixel_acc.avg}')
            print(f'[Testing] Average Pixel Accuracy per Class: {avg_pixel_acc_per_class.avg}')
            print(f'[Testing] Average Jaccard Index: {avg_jacc.avg}')
            print(f'[Testing] Average DICE Coefficient: {avg_dice.avg}')
            print('\n')

            iou, miou = metric.value()
            print('========================')
            print(f'Mean IoU is: {miou}')
            print(f'Class-wise IoU is: {iou[1:]}')
            print('========================')
            print('\n')
            print('\n')

    return np.array(all_decoded_clouds), np.array(all_gt_clouds)



print("Starting testing...")

dec_clouds = f"{export_ds}/sem_clouds_decoded.npy"
dec_gt_clouds = f"{export_ds}/sem_clouds_gt.npy"

torch.cuda.empty_cache()
decoded_clouds, gt_clouds = test_lidarseg(net, criterion, writer)
print(f'Decoded clouds are {decoded_clouds.shape}')
print(f'Decoded gt clouds are {gt_clouds.shape}')

np.save(dec_gt_clouds, gt_clouds)
np.save(dec_clouds, decoded_clouds)
print(f'Wrote decoded spheres to {dec_clouds}.')

writer.close()
print("Testing finished!")

