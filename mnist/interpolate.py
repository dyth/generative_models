#!/usr/bin/env python
"""
linear space interpolation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from tqdm.autonotebook import tqdm
from torchvision.utils import save_image

from dataloaders import *
from convolutional import Decoder

torch.manual_seed(9001)

from fgsm import Autoencoder


for i in range(32):
    for j in range(10):
        interpolate = j * 2 / 9 - 1
        print(interpolate)
