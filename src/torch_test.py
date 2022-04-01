#/usr/bin/env python3

import numpy as np
import torch

print('Running the GPU Test Driver')
print(f'Number of GPUs found: {torch.cuda.device_count()}')
