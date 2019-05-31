#!/usr/bin/env python
"""
train a disentangled convolutional encoder and decoder with variational loss
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
from variational import Encoder, Decoder, Autoencoder

torch.manual_seed(9001)


def variational_loss(output, data, mean, logvar, beta):
    'sum reconstruction and divergence losses'
    reconstruction = F.binary_cross_entropy(output, data, reduction='sum')
    divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction + beta * divergence, reconstruction


def train(model, device, train_loader, optimizer, epoch, beta):
    progress = tqdm(enumerate(train_loader), desc="train", total=len(train_loader))
    model.train()
    train_loss = 0
    for i, (data, _) in progress:
        data = data.to(device)
        optimizer.zero_grad()
        output, mean, logvar = model(data)
        loss, reconstruction = variational_loss(output, data, mean, logvar, beta)
        loss.backward()
        optimizer.step()
        train_loss += reconstruction / np.prod([*data.shape])
        progress.set_description("train loss: {:.4f}".format(train_loss/(i+1)))


def test(model, device, test_loader, folder, epoch, beta):
    progress = tqdm(enumerate(test_loader), desc="test", total=len(test_loader))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in progress:
            data = data.to(device)
            output, mean, logvar = model(data)
            loss, reconstruction = variational_loss(output, data, mean, logvar, beta)
            test_loss += reconstruction / np.prod([*data.shape])
            progress.set_description("test loss: {:.4f}".format(test_loss/(i+1)))
            if i == 0:
                output = output.view(100, 1, 28, 28)
                data = data.view(100, 1, 28, 28)
                save_image(output.cpu(), f'{folder}/{epoch}.png', nrow=10)
                save_image(data.cpu(), f'{folder}/baseline{epoch}.png', nrow=10)



def main():
    beta = 150.0
    batch_size = 64
    test_batch_size = 100
    epochs = 10
    save_model = True
    folder = 'beta-vae'

    if not os.path.exists(folder):
        os.makedirs(folder)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters())

    path = 'data'
    train_loader, test_loader = get_mnist(path, use_cuda, batch_size, test_batch_size)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, beta)
        test(model, device, test_loader, folder, epoch, beta)
        print("")
        if save_model:
            torch.save(model.state_dict(), f"{folder}/{epoch}.pt")



if __name__ == '__main__':
    main()
