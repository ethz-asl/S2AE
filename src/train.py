#!/usr/bin/env python3
import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy import spatial

import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm.auto import tqdm

from data_splitter import DataSplitter
from training_set import TrainingSetLidarSeg
from loss import *
# from model_simple_for_testing import ModelSimpleForTesting
# from model_fcn import ModelFCN
# from model_unet import ModelUnet
from model_segnet import ModelSegnet
from model import Model
from sphere import Sphere
from visualize import Visualize
from metrics import *
from average_meter import AverageMeter

# ## Initialize some parameter
print(f"Initializing CUDA...")
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

print(f"Setting parameters...")
bandwidth = 100
learning_rate = 1e-3
n_epochs = 10
batch_size = 5
num_workers = 32
n_classes = 9

print(f"Initializing data structures...")
# net = ModelSimpleForTesting(bandwidth=bandwidth, n_classes=n_classes).cuda()
# net = ModelUnet(bandwidth=bandwidth, n_classes=n_classes).cuda()
# net = ModelSegnet(bandwidth=bandwidth, n_classes=n_classes).cuda()
net = Model(bandwidth=bandwidth, n_classes=n_classes).cuda()

#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# criterion = L2Loss(alpha=0.5, margin=0.2)
# criterion = CrossEntropyLoss(n_classes=n_classes)
# criterion = NegativeLogLikelihoodLoss(n_classes=n_classes)
criterion = MainLoss()

writer = SummaryWriter()
model_save = 'test_training_params.pkl'

print(f"All instances initialized.")


# ## Load the dataset

# export_ds = '/mnt/data/datasets/nuscenes/processed'
export_ds = '/media/scratch/berlukas/nuscenes'

# training
img_filename = f"{export_ds}/images.npy"
# cloud_filename = f"{export_ds}/clouds1.npy"
cloud_filename = f"{export_ds}/sampled_clouds_fixed_sem.npy"
# sem_clouds_filename = f"{export_ds}/new_sem_classes_gt1.npy"

# testing
dec_input = f"{export_ds}/decoded_input.npy"
dec_clouds = f"{export_ds}/decoded.npy"
dec_gt = f"{export_ds}/decoded_gt.npy"

print(f"Loading clouds from {cloud_filename}")
#img_features = np.load(img_filename)
# img_features = np.zeros((1,1,1))
cloud_features = np.load(cloud_filename)
# sem_cloud_features = np.load(sem_clouds_filename)

sem_cloud_features = cloud_features[:, 2, :, :]
cloud_features = cloud_features[:, 0:2, :, :]
print(f"Shape of clouds is {cloud_features.shape} and sem clouds is {sem_cloud_features.shape}")


# training 2
# cloud_filename = f"{export_ds}/clouds2.npy"
# sem_clouds_filename = f"{export_ds}/new_sem_classes_gt2.npy"

# cloud_features_2 = np.load(cloud_filename)
# sem_cloud_features_2 = np.load(sem_clouds_filename)
# print(f"Shape of clouds (2) is {cloud_features_2.shape} and sem clouds (2) is {sem_cloud_features_2.shape}")

# training 3
# cloud_filename = f"{export_ds}/clouds3.npy"
# sem_clouds_filename = f"{export_ds}/new_sem_classes_gt3.npy"

# cloud_features_3 = np.load(cloud_filename)
# sem_cloud_features_3 = np.load(sem_clouds_filename)
# print(f"Shape of clouds (3) is {cloud_features_3.shape} and sem clouds (3) is {sem_cloud_features_3.shape}")

# cloud_features = np.concatenate((cloud_features, cloud_features_2, cloud_features_3))
# sem_cloud_features = np.concatenate((sem_cloud_features, sem_cloud_features_2, sem_cloud_features_3))

# print(f"Shape of the final clouds is {cloud_features.shape} and sem clouds is {sem_cloud_features.shape}")

#n_process = 30
#img_features = img_features[0:n_process, :, :, :]
#cloud_features = cloud_features[0:n_process, :, :, :]
#sem_cloud_features = sem_cloud_features[0:n_process, :, :]
#print(f"Shape of images is {img_features.shape}, clouds is {cloud_features.shape} and sem clouds is {sem_cloud_features.shape}")


# Initialize the data loaders
train_set = TrainingSetLidarSeg(bandwidth, cloud_features, sem_cloud_features)
print(f"Total size of the training set: {len(train_set)}")
split = DataSplitter(train_set, False, test_train_split=0.95, val_train_split=0.05, shuffle=True)

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

def train_lidarseg(net, criterion, optimizer, writer, epoch, n_iter, loss_, t0):
    net.train()
    for batch_idx, (cloud, lidarseg_gt) in enumerate(train_loader):
        cloud, lidarseg_gt = cloud.cuda().float(), lidarseg_gt.cuda().long()

        enc_dec_cloud = net(cloud)
        loss, loss_total = criterion(enc_dec_cloud, lidarseg_gt)
        #loss_embedd = embedded_a.norm(2) + embedded_p.norm(2) + embedded_n.norm(2)
        #loss = loss_triplet + 0.001 * loss_embedd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ += loss_total.item()

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
            loss, loss_total = criterion(enc_dec_cloud, lidarseg_gt)
            writer.add_scalar('Validation/Loss', loss, n_iter)

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
    return n_iter

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

    return all_input_clouds, all_decoded_clouds, all_gt_clouds


# ## Training Loop

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

print("Training finished!")
torch.save(net.state_dict(), model_save)


# ## Testing

print("Starting testing...")

torch.cuda.empty_cache()
input_clouds, decoded_clouds, gt_clouds = test_lidarseg(net, criterion, writer)

np.save(dec_input, input_clouds)
np.save(dec_gt, gt_clouds)
np.save(dec_clouds, decoded_clouds)

writer.close()
print("Testing finished!")
