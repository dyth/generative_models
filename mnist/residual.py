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

from dataloaders import *

torch.manual_seed(9001)


class BasicBlock(torch.nn.Module):

    def __init__(self, filters=64):
        'residual basic block'
        super(BasicBlock, self).__init__()
        self.residual = torch.nn.Sequential(
            torch.nn.Conv2d(filters, filters, 3, 1, padding=1, bias=False),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters, 3, 1, padding=1, bias=False),
            torch.nn.BatchNorm2d(filters)
        )

    def forward(self, x):
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



class Encoder(nn.Module):

    def __init__(self):
        'define four layers'
        super(Encoder, self).__init__()
        # self.encoder = torch.nn.Sequential(
        self.conv1 = nn.Conv2d(1, 2, 3, 1, padding=1)
        self.conv2 = BasicBlock(2)
        # self.conv2 = nn.Conv2d(2, 2, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(2, 4, 4, 2)
        self.conv4 = BasicBlock(4)
        # self.conv4 = nn.Conv2d(4, 4, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(4, 8, 4, 2)
        self.conv6 = BasicBlock(8)
        # self.conv6 = nn.Conv2d(8, 8, 3, 1, padding=1)
        self.conv7 = nn.Conv2d(8, 16, 4, 2, bias=False)

    def forward(self, x):
        'convolution'
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        return x


class Decoder(nn.Module):

    def __init__(self):
        'define four layers'
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(16, 8, 4, 2, output_padding=1)
        self.conv2 = BasicBlock(8)
        # self.conv2 = nn.Conv2d(8, 8, 3, 1, padding=1)
        self.conv3 = nn.ConvTranspose2d(8, 4, 4, 2, output_padding=1)
        self.conv4 = BasicBlock(4)
        # self.conv4 = nn.Conv2d(4, 4, 3, 1, padding=1)
        self.conv5 = nn.ConvTranspose2d(4, 2, 4, 2)
        self.conv6 = BasicBlock(2)
        # self.conv6 = nn.Conv2d(2, 2, 3, 1, padding=1)
        self.conv7 = nn.Conv2d(2, 1, 3, 1, padding=1)

    def forward(self, x):
        'deconvolution'
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = torch.sigmoid(self.conv7(x))
        return x


class Autoencoder(nn.Module):

    def __init__(self):
        'define encoder and decoder'
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

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
                output = output.view(100, 1, 28, 28)
                data = data.view(100, 1, 28, 28)
                save_image(output.cpu(), f'{folder}/{epoch}.png', nrow=10)
                save_image(data.cpu(), f'{folder}/{epoch}baseline.png', nrow=10)



def main():
    batch_size = 64
    test_batch_size = 100
    epochs = 10
    save_model = True
    folder = 'residual'

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
