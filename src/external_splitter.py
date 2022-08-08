import logging

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class ExternalSplitter:

    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.train_dataset_size = len(train_dataset)
        self.val_dataset = val_dataset
        self.val_dataset_size = len(val_dataset)
        
        self.train_indices = list(range(self.train_dataset_size))
        self.val_indices = list(range(self.val_dataset_size))

        self.train_sampler = None
        self.val_sampler = None

        self.train_loader = None
        self.val_loader = None
        
        self.split_for_train()

    def split_for_train(self):
        np.random.shuffle(self.train_indices)

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)

    def get_split(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train-validation-test dataloaders')

        if self.train_sampler != None: 
            self.train_loader = self.get_train_loader(
                batch_size=batch_size, num_workers=num_workers)
        if self.val_sampler != None:
            self.val_loader = self.get_validation_loader(
                batch_size=batch_size, num_workers=num_workers)

        return self.train_loader, self.val_loader

    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
        return self.train_loader

    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    def get_train_size(self):
        return len(self.train_indices)

    def get_val_size(self):
        return len(self.val_indices)
