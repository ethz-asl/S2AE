#!/usr/bin/env python
# coding: utf-8

# # Training code for Fusion Network of S2AE

import math
import time

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data_splitter import DataSplitter
from training_set import TrainingSetFusedSeg
from model_fused import FusedModel
from average_meter import AverageMeter

from metrics import *
from loss import *

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
n_classes = 17

print(f"Initializing data structures...")
net = FusedModel(bandwidth=bandwidth, n_classes=n_classes)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = MainLoss()

writer = SummaryWriter()
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
model_save = f'fused_model_{timestamp}'
print(f"All instances initialized.")

# ## Load the dataset

# export_ds = '/mnt/data/datasets/nuscenes/processed'
export_ds = '/media/scratch/berlukas/nuscenes'
# export_ds = '/cluster/work/riner/users/berlukas'

# training
img_filename = f"{export_ds}/color_images_150_400.npy"
cloud_filename = f"{export_ds}/sem_clouds_16_400.npy"
sem_clouds_filename = f"{export_ds}/sem_clouds_decoded_400.npy"

# testing
dec_input_clouds = f"{export_ds}/decoded_fused_input_clouds.npy"
dec_input_images = f"{export_ds}/decoded_fused_input_images.npy"
dec_clouds = f"{export_ds}/decoded_fused.npy"
dec_gt = f"{export_ds}/decoded_fused_gt.npy"

print(f"Loading from images from {img_filename}, clouds from {cloud_filename} and sem clouds from {sem_clouds_filename}")
img_features = np.load(img_filename)
print('Loaded images.')
cloud_features = np.load(cloud_filename)
# cloud_features = cloud_features[:, 2, :, :]
print('Loaded clouds.')
sem_cloud_features = np.load(sem_clouds_filename)
print('Loaded decoded.')

print(f"Shape of images is {img_features.shape}, clouds is {cloud_features.shape} and sem clouds is {sem_cloud_features.shape}")


# --- EXTERNAL SPLITTING ---------------------------------------------
gt_val_filename = f"{export_ds}/sem_clouds_val_16.npy"
img_filename = f"{export_ds}/color_images_val.npy"
decoded_filename = f"{export_ds}/sem_clouds_val_decoded.npy"

print(f"Loading clouds from {gt_val_filename}.")
cloud_val = np.load(gt_val_filename)
print(f"Loading images from {img_filename}.")
img_val = np.load(img_filename)
print(f"Loading decoded from {decoded_filename}.")
decoded_val = np.load(decoded_filename)

sem_val_features = np.copy(cloud_val[:, 2, :, :])
val_features = cloud_val[:, 0:2, :, :]
print(f"Shape clouds is {val_features.shape} and sem clouds is {sem_val_features.shape}")

train_set = TrainingSetFusedSeg(sem_cloud_features, img_features, cloud_features)
val_set = TrainingSetFusedSeg(val_features, sem_val_features)
split = ExternalSplitter(train_set, val_set)
train_loader, val_loader = split.get_split(batch_size=batch_size, num_workers=num_workers)
train_size = split.get_train_size()
val_size = split.get_val_size()
test_size = 0
# --------------------------------------------------------------------



# Initialize the data loaders
# train_set = TrainingSetFusedSeg(sem_cloud_features, img_features, cloud_features)
# print(f"Total size of the training set: {len(train_set)}")
# split = DataSplitter(train_set, False, test_train_split=0.95, shuffle=True)

# Split the data into train, val and optionally test
# train_loader, val_loader, test_loader = split.get_split(
#     batch_size=batch_size, num_workers=num_workers)
# train_size = split.get_train_size()
# val_size = split.get_val_size()
# test_size = split.get_test_size()


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
        #loss_embedd = embedded_a.norm(2) + embedded_p.norm(2) + embedded_n.norm(2)
        #loss = loss_triplet + 0.001 * loss_embedd

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
            writer.add_scalar('Validation/Loss', float(loss), n_iter)

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
    save_checkpoint(net, optimizer, criterion, lr, epoch)

print("Training finished!")
torch.save(net.state_dict(), model_save)

# ## Testing
if test_size > 0:
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
