#!/usr/bin/env python
"""
linear space interpolation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import copy
import numpy as np
from tqdm.autonotebook import tqdm
from torchvision.utils import save_image

from dataloaders import *
from convolutional import Decoder

torch.manual_seed(9001)

from fgsm import Autoencoder


def save_image_as_numpy(use_cuda):
    'get dataloader for test set with batch size = 1'
    batch_size = 64
    test_batch_size = 1
    epochs = 100
    path = 'data'
    _, test_loader = get_mnist(path, use_cuda, batch_size, test_batch_size)
    return test_loader


def traverse_sigmoidal_latent_space(model, batch, device):
    'for every axis in latent space, traverse individually from -1 to 1'
    batch = batch.to(device)
    latent = model.encoder(batch)[0]
    traversed = []
    for j in range(10):
        traverse = []
        for i in range(32):
            t = copy.deepcopy(latent)
            t[i] = j / 9
            traverse.append(t)
        traverse = torch.stack(traverse)
        traverse = traverse.to(device)
        traverse = model.decoder(traverse)
        traversed.append(traverse)
    return traversed


def latent_traversal(test_loader, model, device, folder, i):
    'traverse all 32 axes of the latent representation to produce an image'
    # get batch
    for batch, labels in test_loader:
        break
    # traverse latent space and save image
    traversed = traverse_sigmoidal_latent_space(model, batch, device)
    traversed = torch.stack(traversed)
    traversed = traversed.view(320, 1, 28, 28)
    save_image(traversed.cpu(), f'{folder}/{i}.png', nrow=32)


def main():
    folder = 'interpolate'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(1, 100):
        with torch.no_grad():
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            model = Autoencoder().to(device)
            model.load_state_dict(torch.load(f'fgsm/{i}.pt'))

            test_loader = save_image_as_numpy(use_cuda)
            latent_traversal(test_loader, model, device, folder, i)


if __name__ == '__main__':
    main()
