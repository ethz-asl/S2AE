#!/usr/bin/env python3

import math
import time
import datetime

import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import nvidia_smi

from data_splitter import DataSplitter
from training_set import TrainingSetLidarSeg
from model import Model
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
n_epochs = 10
batch_size = 5
num_workers = 32
n_classes = 9
device_ids = [0]

print(f"Initializing data structures...")
print(f'Training will run on these gpus {device_ids}')
print(f'We have a batch size of {batch_size} and {n_epochs} epochs.')
print(f'We will use {num_workers} workers')
# net = ModelSimpleForTesting(bandwidth=bandwidth, n_classes=n_classes).cuda()
# net = ModelUnet(bandwidth=bandwidth, n_classes=n_classes).cuda()
# net = ModelSegnet(bandwidth=bandwidth, n_classes=n_classes).cuda()
model = Model(bandwidth=bandwidth, n_classes=n_classes).cuda(0)
net = nn.DataParallel(model, device_ids = device_ids).to(0)

#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

criterion = MainLoss()
writer = SummaryWriter()
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
model_save = f'test_lidarseg_{timestamp}'

print('\n')
print(f'All instances initialized.')
print(f'Saving final model to {model_save}')


# ## Load the dataset

# export_ds = '/mnt/data/datasets/nuscenes/processed'
export_ds = '/media/scratch/berlukas/nuscenes'
# export_ds = '/cluster/work/riner/users/berlukas'


# testing
dec_input = f"{export_ds}/decoded_input_lidar.npy"
dec_clouds = f"{export_ds}/decoded_lidar.npy"
dec_gt = f"{export_ds}/decoded_gt_lidar.npy"

# training
cloud_filename = f"{export_ds}/sem_clouds.npy"
print(f"Loading clouds from {cloud_filename}.")
cloud_features = np.load(cloud_filename)
# cloud_filename = f"{export_ds}/sem_clouds_100_200.npy"


# --- DATA MERGING ---------------------------------------------------
cloud_filename_2 = f"{export_ds}/sem_clouds2.npy"
# cloud_filename_3 = f"{export_ds}/sem_clouds3.npy"

cloud_features_2 = np.load(cloud_filename_2)
# cloud_features_3 = np.load(cloud_filename_3)
print(f"Shape of sem clouds 1 is {cloud_features.shape}")
print(f"Shape of sem clouds 2 is {cloud_features_2.shape}")
# print(f"Shape of sem clouds 3 is {cloud_features_3.shape}")
cloud_features = np.concatenate((cloud_features, cloud_features_2))
# cloud_features = np.concatenate((cloud_features, cloud_features_2, cloud_features_3))
# --------------------------------------------------------------------

# --- TEST TRAINING --------------------------------------------------
# n_process = 200
# cloud_features = cloud_features[0:n_process, :, :, :]
# --------------------------------------------------------------------

# --- DATA SPLITTING -------------------------------------------------
# sem_cloud_features = np.copy(cloud_features[:, 2, :, :])
# cloud_features = cloud_features[:, 0:2, :, :]
# print(f"Shape clouds is {cloud_features.shape} and sem clouds is {sem_cloud_features.shape}")

# # Initialize the data loaders
# train_set = TrainingSetLidarSeg(cloud_features, sem_cloud_features)
# print(f"Total size of the training set: {len(train_set)}")
# split = DataSplitter(train_set, False, test_train_split=1.0, val_train_split=0.05, shuffle=True)

# # Split the data into train, val and optionally test
# train_loader, val_loader, test_loader = split.get_split(
#     batch_size=batch_size, num_workers=num_workers)
# train_size = split.get_train_size()
# val_size = split.get_val_size()
# test_size = split.get_test_size()
# --------------------------------------------------------------------

# --- EXTERNAL SPLITTING ---------------------------------------------
val_filename = f"{export_ds}/sem_clouds_val.npy"

print(f"Loading clouds from {val_filename}.")
cloud_val = np.load(val_filename)

sem_val_features = np.copy(cloud_val[:, 2, :, :])
val_features = cloud_val[:, 0:2, :, :]
print(f"Shape clouds is {val_features.shape} and sem clouds is {sem_val_features.shape}")

train_set = TrainingSetLidarSeg(cloud_features, sem_cloud_features)
val_set = TrainingSetLidarSeg(val_features, sem_val_features)
split = ExternalSplitter(train_set, val_set)
train_loader, val_loader = split.get_split(batch_size=batch_size, num_workers=num_workers)
train_size = split.get_train_size()
val_size = split.get_val_size()
# --------------------------------------------------------------------


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

def train_lidarseg(net, criterion, optimizer, writer, epoch, n_iter, loss_, t0):
    net.train()
    for batch_idx, (cloud, lidarseg_gt) in enumerate(train_loader):
        cloud, lidarseg_gt = cloud.cuda().float(), lidarseg_gt.cuda().long()
        
        enc_dec_cloud = net(cloud)
        loss = criterion(enc_dec_cloud, lidarseg_gt)

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

def validate_lidarseg(net, criterion, optimizer, writer, epoch, n_iter):
    avg_pixel_acc = AverageMeter()
    avg_pixel_acc_per_class = AverageMeter()
    avg_jacc = AverageMeter()
    avg_dice = AverageMeter()
    net.eval()
    with torch.no_grad():
        for batch_idx, (cloud, lidarseg_gt) in enumerate(val_loader):
            cloud, lidarseg_gt = cloud.cuda().float(), lidarseg_gt.cuda().long()
            enc_dec_cloud = net(cloud)

            optimizer.zero_grad()
            loss = criterion(enc_dec_cloud, lidarseg_gt)
            writer.add_scalar('Validation/Loss', loss.mean().item(), n_iter)

            pred_segmentation = torch.argmax(enc_dec_cloud, dim=1)
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

        print(f'[Validation for epoch {epoch_p_1}] Average Pixel Accuracy: {avg_pixel_acc.avg}')
        print(f'[Validation for epoch {epoch_p_1}] Average Pixel Accuracy per Class: {avg_pixel_acc_per_class.avg}')
        print(f'[Validation for epoch {epoch_p_1}] Average Jaccard Index: {avg_jacc.avg}')
        print(f'[Validation for epoch {epoch_p_1}] Average DICE Coefficient: {avg_dice.avg}')
        print('\n')
    return n_iter
   
def save_checkpoint(net, optimizer, criterion, lr, n_epoch):
    checkpoint_path = f'./checkpoints/{model_save}_{n_epoch}.pth'
    torch.save({
            'epoch': n_epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            'lr': lr,
            }, checkpoint_path)
    print('================================')
    print(f'Saved checkpoint to {checkpoint_path}')
    print('================================')

def test_lidarseg(net, criterion, writer):
    all_input_clouds = [None] * test_size
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
        for batch_idx, (cloud, lidarseg_gt) in enumerate(test_loader):
            cloud, lidarseg_gt = cloud.cuda().float(), lidarseg_gt.cuda().long()
            enc_dec_cloud = net(cloud)

            pred_segmentation = torch.argmax(enc_dec_cloud, dim=1)
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
                all_input_clouds[k] = cloud.cpu().data.numpy()[i,:,:,:]
                all_decoded_clouds[k] = enc_dec_cloud.cpu().data.numpy()[i,:,:,:]
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

    return all_input_clouds, all_decoded_clouds, all_gt_clouds


# Training Loop

abort = False
train_iter = 0
val_iter = 0
loss_ = 0.0
print(f'Starting training using {n_epochs} epochs')
for epoch in tqdm(range(n_epochs)):    
    lr = adjust_learning_rate_exp(optimizer, epoch_num=epoch, lr=learning_rate)
    t0 = time.time()

    train_iter = train_lidarseg(net, criterion, optimizer, writer, epoch, train_iter, loss_, t0)    
    val_iter = validate_lidarseg(net, criterion, optimizer, writer, epoch, val_iter)
    writer.add_scalar('Train/lr', lr, epoch)
    save_checkpoint(net, optimizer, criterion, lr, epoch)


print("Training finished!")
final_save_path = f'./{model_save}.pkl'
torch.save(net.state_dict(), final_save_path)
print(f'Saved final weights to {final_save_path}.')

# Show GPU Utilization
nvidia_smi.nvmlInit()
deviceCount = nvidia_smi.nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
nvidia_smi.nvmlShutdown()


# Testing
if test_size > 0:
    print("Starting testing...")
    torch.cuda.empty_cache()
    input_clouds, decoded_clouds, gt_clouds = test_lidarseg(net, criterion, writer)

    np.save(dec_input, input_clouds)
    np.save(dec_gt, gt_clouds)
    np.save(dec_clouds, decoded_clouds)

    writer.close()
    print("Testing finished!")
