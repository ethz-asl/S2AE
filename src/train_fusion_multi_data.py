#!/usr/bin/env python
# coding: utf-8

# # Training code for Fusion Network of S2AE

import math
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import nvidia_smi

from data_splitter import DataSplitter
from training_set import TrainingSetFusedSeg
from model_fused import FusedModel
from average_meter import AverageMeter

from metrics import *
from loss import *

# ## Initialize some parameter

print(f"Initializing CUDA...")
# torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

print(f"Setting parameters...")
bandwidth = 100
learning_rate = 1e-3
n_epochs = 4
batch_size = 10
num_workers = 20
n_classes = 9
device_ids = [0, 1, 2, 3, 4]

print(f"Initializing data structures...")
print(f'Training will run on these gpus {device_ids}')
print(f'We have a batch size of {batch_size} and {n_epochs} epochs.')
print(f'We will use {num_workers} workers')
model = FusedModel(bandwidth=bandwidth, n_classes=n_classes).cuda(0)
net = nn.DataParallel(model, device_ids = device_ids).to(0)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = MainLoss()

writer = SummaryWriter()
model_save = 'fused_model_1.pkl'

print(f"All instances initialized.")


# ## Load the dataset


# export_ds = '/mnt/data/datasets/nuscenes/processed'
export_ds = '/media/scratch/berlukas/nuscenes'
export_ds = '/cluster/work/riner/users/berlukas'

# training
img_filename = f"{export_ds}/color_images_150.npy"
cloud_filename = f"{export_ds}/sem_clouds.npy"
sem_clouds_filename = f"{export_ds}/sem_clouds_decoded.npy"

# testing
dec_input_clouds = f"{export_ds}/decoded_fused_input_clouds.npy"
dec_input_images = f"{export_ds}/decoded_fused_input_images.npy"
dec_clouds = f"{export_ds}/decoded_fused.npy"
dec_gt = f"{export_ds}/decoded_fused_gt.npy"

print(f"Loading from images from {img_filename}, clouds from {cloud_filename} and sem clouds from {sem_clouds_filename}")
img_features = np.load(img_filename)
print('Loaded images.')
cloud_features = np.load(cloud_filename)
cloud_features = cloud_features[:, 2, :, :]
print('Loaded clouds.')
sem_cloud_features = np.load(sem_clouds_filename)
print('Loaded decoded.')

print(f"Shape of images is {img_features.shape}, clouds is {cloud_features.shape} and sem clouds is {sem_cloud_features.shape}")



# Initialize the data loaders
train_set = TrainingSetFusedSeg(sem_cloud_features, img_features, cloud_features)
print(f"Total size of the training set: {len(train_set)}")
split = DataSplitter(train_set, False, test_train_split=0.95, shuffle=True)

# Split the data into train, val and optionally test
train_loader, val_loader, test_loader = split.get_split(
    batch_size=batch_size, num_workers=num_workers)
train_size = split.get_train_size()
val_size = split.get_val_size()
test_size = split.get_test_size()


print("Training size: ", train_size)
print("Validation size: ", val_size)
if test_size == 0:
    print('Test size is 0. Configured for external tests')
else:
    print("Testing size: ", test_size)



def adjust_learning_rate_exp(optimizer, epoch_num, lr):
    decay_rate = 0.96
    new_lr = lr * math.pow(decay_rate, epoch_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr

def train_fused_lidarseg(net, criterion, optimizer, writer, epoch, n_iter, loss_, t0):
    net.train()
    for batch_idx, (decoded, image, lidarseg_gt) in enumerate(train_loader):
        decoded, image, lidarseg_gt = decoded.cuda().float(), image.cuda().float(), lidarseg_gt.cuda().long()

        enc_fused_dec = net(decoded, image)
        loss = criterion(enc_fused_dec, lidarseg_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ += loss.mean().item()

        writer.add_scalar('Train/Loss', loss, n_iter)
        n_iter += 1

        if batch_idx % 100 == 99:
            t1 = time.time()
            print('[Epoch %d, Batch %4d] loss: %.8f time: %.5f lr: %.3e' %
                  (epoch + 1, batch_idx + 1, loss_ / 100, (t1 - t0) / 60, lr))
            t0 = t1
            loss_ = 0.0
    return n_iter

def validate_fused_lidarseg(net, criterion, optimizer, writer, epoch, n_iter):
    avg_pixel_acc = AverageMeter()
    avg_pixel_acc_per_class = AverageMeter()
    avg_jacc = AverageMeter()
    avg_dice = AverageMeter()
    net.eval()
    with torch.no_grad():
        for batch_idx, (decoded, image, lidarseg_gt) in enumerate(val_loader):
            decoded, image, lidarseg_gt = decoded.cuda().float(), image.cuda().float(), lidarseg_gt.cuda().long()
            enc_fused_dec = net(decoded, image)

            optimizer.zero_grad()
            loss = criterion(enc_fused_dec, lidarseg_gt)
            writer.add_scalar('Validation/Loss', loss, n_iter)

            pred_segmentation = torch.argmax(enc_fused_dec, dim=1)
            pixel_acc, pixel_acc_per_class, jacc, dice = eval_metrics(lidarseg_gt, pred_segmentation, num_classes = n_classes)
            avg_pixel_acc.update(pixel_acc)
            avg_pixel_acc_per_class.update(pixel_acc_per_class)
            avg_jacc.update(jacc)
            avg_dice.update(dice)

            n_iter += 1

        epoch_p_1 = epoch+1
        writer.add_scalar('Validation/AvgPixelAccuracy', avg_pixel_acc.avg, epoch_p_1)
        writer.add_scalar('Validation/AvgPixelAccuracyPerClass', avg_pixel_acc_per_class.avg, epoch_p_1)
        writer.add_scalar('Validation/AvgJaccardIndex', avg_jacc.avg, epoch_p_1)
        writer.add_scalar('Validation/AvgDiceCoefficient', avg_dice.avg, epoch_p_1)
    return n_iter

def test_fused_lidarseg(net, criterion, writer):
    all_input_clouds = [None] * test_size
    all_input_images = [None] * test_size
    all_decoded_clouds = [None] * test_size
    all_gt_clouds = [None] * test_size
    k = 0
    avg_pixel_acc = AverageMeter()
    avg_pixel_acc_per_class = AverageMeter()
    avg_jacc = AverageMeter()
    avg_dice = AverageMeter()
    n_iter = 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (decoded, image, lidarseg_gt) in enumerate(test_loader):
            decoded, image, lidarseg_gt = decoded.cuda().float(), image.cuda().float(), lidarseg_gt.cuda().long()
            enc_fused_dec = net(decoded, image)

            pred_segmentation = torch.argmax(enc_fused_dec, dim=1)
            pixel_acc, pixel_acc_per_class, jacc, dice = eval_metrics(lidarseg_gt, pred_segmentation, num_classes = n_classes)
            avg_pixel_acc.update(pixel_acc)
            avg_pixel_acc_per_class.update(pixel_acc_per_class)
            avg_jacc.update(jacc)
            avg_dice.update(dice)

            writer.add_scalar('Test/PixelAccuracy', pixel_acc, n_iter)
            writer.add_scalar('Test/PixelAccuracyPerClass', pixel_acc_per_class, n_iter)
            writer.add_scalar('Test/JaccardIndex', jacc, n_iter)
            writer.add_scalar('Test/DiceCoefficient', dice, n_iter)

            n_batch = enc_fused_dec.shape[0]
            for i in range(0, n_batch):
                all_input_clouds[k] = decoded.cpu().data.numpy()[i,:,:,:]
                all_input_images[k] = image.cpu().data.numpy()[i,:,:,:]
                all_decoded_clouds[k] = enc_fused_dec.cpu().data.numpy()[i,:,:,:]
                all_gt_clouds[k] = lidarseg_gt.cpu().data.numpy()[i,:,:]
                k = k + 1
            n_iter += 1

        writer.add_scalar('Test/AvgPixelAccuracy', avg_pixel_acc.avg, n_iter)
        writer.add_scalar('Test/AvgPixelAccuracyPerClass', avg_pixel_acc_per_class.avg, n_iter)
        writer.add_scalar('Test/AvgJaccardIndex', avg_jacc.avg, n_iter)
        writer.add_scalar('Test/AvgDiceCoefficient', avg_dice.avg, n_iter)

        print(f'Average Pixel Accuracy: {avg_pixel_acc.avg}')
        print(f'Average Pixel Accuracy per Class: {avg_pixel_acc_per_class.avg}')
        print(f'Average Jaccard Index: {avg_jacc.avg}')
        print(f'Average DICE Coefficient: {avg_dice.avg}')

    return all_input_clouds, all_input_images, all_decoded_clouds, all_gt_clouds


# ## Training Loop


abort = False
train_iter = 0
val_iter = 0
loss_ = 0.0
print(f'Starting training using {n_epochs} epochs')
for epoch in tqdm(range(n_epochs)):
    lr = adjust_learning_rate_exp(optimizer, epoch_num=epoch, lr=learning_rate)
    t0 = time.time()

    train_iter = train_fused_lidarseg(net, criterion, optimizer, writer, epoch, train_iter, loss_, t0)
    val_iter = validate_fused_lidarseg(net, criterion, optimizer, writer, epoch, val_iter)
    writer.add_scalar('Train/lr', lr, epoch)

print("Training finished!")
torch.save(net.state_dict(), model_save)

# Show GPU Utilization
nvidia_smi.nvmlInit()
deviceCount = nvidia_smi.nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
nvidia_smi.nvmlShutdown()

# ## Testing


print("Starting testing...")

torch.cuda.empty_cache()
input_clouds, input_images, decoded_clouds, gt_clouds = test_fused_lidarseg(net, criterion, writer)

np.save(dec_input_clouds, input_clouds)
np.save(dec_input_images, input_images)
np.save(dec_clouds, decoded_clouds)
np.save(dec_gt, gt_clouds)
print(f'Wrote input clouds to {dec_input_clouds}.')
print(f'Wrote input images to {dec_input_images}.')
print(f'Wrote upsampled decoded clouds to {dec_clouds}')
print(f'Wrote upsampled gt clouds to {dec_gt}')

writer.close()
print("Testing finished!")
