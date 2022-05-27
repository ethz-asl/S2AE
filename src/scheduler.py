#!/bin/env python3 
import numpy as np

def cosine_schedule_with_warmup(k, num_epochs, n_gpus, batch_size, dataset_size):
    batch_size *= n_gpus

    if n_gpus == 1:
        warmup_iters = 0
    else:
        warmup_iters = 1000 // n_gpus

    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
        ratio = (k - warmup_iters) / (num_epochs * iter_per_epoch)
        return 0.5 * (1 + np.cos(np.pi * ratio))
