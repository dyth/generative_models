#!/usr/bin/env python
"""
train a convolutional encoder and decoder with variational loss
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


class Encoder(nn.Module):

    def __init__(self):
        'define four layers'
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, 2)
        self.fc1 = nn.Linear(256, 96)
        self.mean = nn.Linear(96, 32)
        self.logvar = nn.Linear(96, 32)

    def forward(self, x):
        'convolution'
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        return F.relu(self.mean(x)), F.relu(self.logvar(x))


class Autoencoder(nn.Module):

    def __init__(self):
        'define encoder and decoder'
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def repameterise(self, mean, logvar):
        'sample encoding from gaussian of mean and logvar'
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            sample = mean + eps*std
            return sample
        else:
            return mean

    def forward(self, x):
        'pass through encoder and decoder'
        mean, logvar = self.encoder(x)
        x = self.repameterise(mean, logvar)
        x = self.decoder(x)
        return x, mean, logvar



def variational_loss(output, data, mean, logvar):
    'sum reconstruction and divergence losses'
    reconstruction = F.binary_cross_entropy(output, data, reduction='sum')
    divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction + divergence, reconstruction


def train(model, device, train_loader, optimizer, epoch):
    progress = tqdm(enumerate(train_loader), desc="train", total=len(train_loader))
    model.train()
    train_loss = 0
    for i, (data, _) in progress:
        data = data.to(device)
        optimizer.zero_grad()
        output, mean, logvar = model(data)
        loss, reconstruction = variational_loss(output, data, mean, logvar)
        loss.backward()
        optimizer.step()
        train_loss += reconstruction / np.prod([*data.shape])
        progress.set_description("train loss: {:.4f}".format(train_loss/(i+1)))


def test(model, device, test_loader, folder, epoch):
    progress = tqdm(enumerate(test_loader), desc="test", total=len(test_loader))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in progress:
            data = data.to(device)
            output, mean, logvar = model(data)
            loss, reconstruction = variational_loss(output, data, mean, logvar)
            test_loss += reconstruction / np.prod([*data.shape])
            progress.set_description("test loss: {:.4f}".format(test_loss/(i+1)))
            if i == 0:
                output = output.view(100, 1, 28, 28)
                data = data.view(100, 1, 28, 28)
                save_image(output.cpu(), f'{folder}/{epoch}.png', nrow=10)
                save_image(data.cpu(), f'{folder}/baseline{epoch}.png', nrow=10)



def main():
    batch_size = 64
    test_batch_size = 100
    epochs = 10
    save_model = True
    folder = 'variational'

    if not os.path.exists(folder):
        os.makedirs(folder)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters())

    path = 'data'
    train_loader, test_loader = get_mnist(path, use_cuda, batch_size, test_batch_size)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, folder, epoch)
        print("")
        if save_model:
            torch.save(model.state_dict(), f"{folder}/{epoch}.pt")



if __name__ == '__main__':
    main()
