#!/usr/bin/env python
# coding: utf-8

# # Test Existing Network

# In[1]:


import math
import sys
import time

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
bandwidth = 100
batch_size = 5
num_workers = 32
n_classes = 7
device_ids = [0, 1, 2, 3, 4]

print(f"Initializing data structures...")
criterion = MainLoss()

writer = SummaryWriter()

model = Model(bandwidth=bandwidth, n_classes=n_classes).cuda(0)
net = nn.DataParallel(model, device_ids = device_ids).to(0)

# chkp = './checkpoints/euler_lidarseg_20220527192933_13.pth'
chkp = './checkpoints/euler_lidarseg_20220527192933_16.pth'

print(f'Loading checkpoint from {chkp}...')
checkpoint = torch.load(chkp)

print('Loading trained model weights...')
net.load_state_dict(checkpoint['model_state_dict'])

print(f"All instances initialized.")


# export_ds = '/mnt/data/datasets/nuscenes/processed'
# export_ds = '/media/scratch/berlukas/nuscenes'
export_ds = '/cluster/work/riner/users/berlukas'

# cloud_filename = f"{export_ds}/sem_clouds_val_400.npy"
cloud_filename = f"{export_ds}/sem_clouds_val_kitti_final_new.npy"

# cloud_filename = f"{export_ds}/sem_clouds.npy"

print(f"Loading clouds from {cloud_filename}.")
cloud_features = np.load(cloud_filename)
cloud_features = cloud_features[0:500, :, :, :]

sem_cloud_features = np.copy(cloud_features[:, 2, :, :])
cloud_features = cloud_features[:, 0:2, :, :]
#cloud_features[:,1,:,:] = np.zeros(cloud_features[:,1,:,:].shape)
print(f"Shape of clouds is {cloud_features.shape} and sem clouds is {sem_cloud_features.shape}")


# Initialize the data loaders
train_set = TrainingSetLidarSeg(cloud_features, sem_cloud_features)
print(f"Total size of the training set: {len(train_set)}")
# split = DataSplitter(train_set, True, test_train_split=0.0, val_train_split=0.0, shuffle=False)
split = ExternalSplitter(train_set, [])

# Split the data into train, val and optionally test
data_loader, _ = split.get_split(batch_size=batch_size, num_workers=num_workers)
data_size = split.get_train_size()

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

    num_classes = 7
    ignore_index = 0
    bw=100
    metric = IoU(num_classes, ignore_index=ignore_index)

    net.eval()
    with torch.no_grad():            
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

                pred = torch.reshape(pred_segmentation[i, :, :], [1, 2*bw, 2*bw]).int()
                gt = torch.reshape(lidarseg_gt[i, :, :], [1, 2*bw, 2*bw]).int()
                metric.add(pred, gt)
            n_iter += 1

            
        writer.add_scalar('Test/AvgPixelAccuracy', avg_pixel_acc.avg, n_iter)   
        writer.add_scalar('Test/AvgPixelAccuracyPerClass', avg_pixel_acc_per_class.avg, n_iter)   
        writer.add_scalar('Test/AvgJaccardIndex', avg_jacc.avg, n_iter)
        writer.add_scalar('Test/AvgDiceCoefficient', avg_dice.avg, n_iter)  

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

