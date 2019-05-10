#!/usr/bin/env python
"""
train a convolutional encoder and decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm.autonotebook import tqdm
from torchvision.utils import save_image

import parts
from dataloaders import *

torch.manual_seed(9001)


class BasicBlock(torch.nn.Module):

    def __init__(self, filters=64):
        'residual basic block'
        super(BasicBlock, self).__init__()
        self.residual = torch.nn.Sequential(
            torch.nn.Conv2d(filters, filters, 3, 1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(filters),
            torch.nn.Conv2d(filters, filters, 3, 1, padding=1, bias=False),
            torch.nn.BatchNorm2d(filters)
        )

    def forward(self, x):
        'output = residual + skip'
        return x + self.residual(x)


class ELU_BatchNorm2d(torch.nn.Module):

    def __init__(self, filters=64):
        super(ELU_BatchNorm2d, self).__init__()
        self.actnorm = torch.nn.Sequential(
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(filters)
        )

    def forward(self, x):
        return self.actnorm(x)


class ResidualEncoder(torch.nn.Module):

    def __init__(self):
        'define four layers'
        super(ResidualEncoder, self).__init__()
        self.activate = torch.nn.ELU()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, 1, padding=1, bias=False),
            ELU_BatchNorm2d(64),

            # BasicBlock(64),
            # ELU_BatchNorm2d(64),
            # torch.nn.Conv2d(64, 64, 1, 1),
            # self.activate,

            BasicBlock(64),
            ELU_BatchNorm2d(64),
            # self.activate,
            # BasicBlock(64),
            # ELU_BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 3, 2),
            self.activate,

            # BasicBlock(128),
            # ELU_BatchNorm2d(128),
            # torch.nn.Conv2d(128, 128, 1, 1),
            # self.activate,

            BasicBlock(128),
            ELU_BatchNorm2d(128),
            # self.activate,
            # BasicBlock(128),
            # ELU_BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 3, 2),
            self.activate,

            BasicBlock(256),
            ELU_BatchNorm2d(256),
            torch.nn.Conv2d(256, 256, 3, 2),
            self.activate,

            BasicBlock(256),
            ELU_BatchNorm2d(256),
            torch.nn.Conv2d(256, 256, 3, 2)
        )

    def forward(self, x):
        return self.encoder(x)


class ResidualDecoder(torch.nn.Module):

    def __init__(self):
        'define four layers'
        super(ResidualDecoder, self).__init__()
        self.activate = torch.nn.ELU()

        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ConvTranspose2d(256, 256, 3, 2),
            ELU_BatchNorm2d(256),
            BasicBlock(256),
            self.activate,

            torch.nn.ConvTranspose2d(256, 256, 3, 2),
            ELU_BatchNorm2d(256),
            BasicBlock(256),
            self.activate,

            torch.nn.ConvTranspose2d(256, 128, 3, 2),
            ELU_BatchNorm2d(128),
            BasicBlock(128),
            # ELU_BatchNorm2d(128),
            # self.activate,
            # BasicBlock(128),
            self.activate,

            # torch.nn.Conv2d(128, 128, 1, 1),
            # ELU_BatchNorm2d(128),
            # BasicBlock(128),
            # self.activate,

            torch.nn.ConvTranspose2d(128, 64, 3, 2, output_padding=1),
            # ELU_BatchNorm2d(64),
            # BasicBlock(64),
            # self.activate,
            ELU_BatchNorm2d(64),
            BasicBlock(64),
            self.activate,

            # torch.nn.Conv2d(64, 64, 1, 1),
            # ELU_BatchNorm2d(64),
            # BasicBlock(64),
            # self.activate,

            torch.nn.Conv2d(64, 3, 3, 1, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self):
        'define encoder and decoder'
        super(Autoencoder, self).__init__()
        self.encoder = ResidualEncoder()
        self.decoder = ResidualDecoder()

    def forward(self, x):
        'pass through encoder and decoder'
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def train(model, device, train_loader, optimizer, epoch):
    progress = tqdm(enumerate(train_loader), desc="train", total=len(train_loader))
    model.train()
    train_loss = 0
    for i, (data, _) in progress:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        train_loss += loss
        progress.set_description("train loss: {:.4f}".format(train_loss/(i+1)))


def test(model, device, test_loader, folder, epoch):
    progress = tqdm(enumerate(test_loader), desc="test", total=len(test_loader))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in progress:
            data = data.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, data)
            progress.set_description("test loss: {:.4f}".format(test_loss/(i+1)))
            if i == 0:
                output = output.view(100, 3, 32, 32)
                data = data.view(100, 3, 32, 32)
                save_image(output.cpu(), f'{folder}/{epoch}.png', nrow=10)
                save_image(data.cpu(), f'{folder}/{epoch}baseline.png', nrow=10)



def main():
    batch_size = 64
    test_batch_size = 100
    epochs = 100
    save_model = True
    folder = 'residual_cifar'

    if not os.path.exists(folder):
        os.makedirs(folder)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters())

    path = 'data'
    train_loader, test_loader = get_cifar10(path, use_cuda, batch_size, test_batch_size)

    for epoch in range(1, epochs + 1):
        print(f"\n{epoch}")
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, folder, epoch)
        if save_model:
            torch.save(model.state_dict(), f"{folder}/{epoch}.pt")



if __name__ == '__main__':
    main()
