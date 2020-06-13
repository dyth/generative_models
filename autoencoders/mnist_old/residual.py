#!/usr/bin/env python
"""
train a residual encoder and decoder
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


class Encoder(nn.Module):

    def __init__(self, filters):
        'define four layers'
        super(Encoder, self).__init__()
        self.filters = filters
        self.encoder = nn.Sequential(
            nn.Conv2d(1, filters[0], 3, 1, padding=1),

            nn.BatchNorm2d(filters[0]),
            BasicBlock(filters[0]),
            nn.Conv2d(filters[0], filters[1], 4, 2),

            nn.BatchNorm2d(filters[1]),
            BasicBlock(filters[1]),
            nn.Conv2d(filters[1], filters[2], 4, 2),

            nn.BatchNorm2d(filters[2]),
            BasicBlock(filters[2]),
            nn.Conv2d(filters[2], filters[3], 4, 2, bias=False)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):

    def __init__(self, filters):
        'define four layers'
        super(Decoder, self).__init__()
        self.filters = filters
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[2], 4, 2, output_padding=1),
            BasicBlock(filters[2]),
            nn.BatchNorm2d(filters[2]),

            nn.ConvTranspose2d(filters[2], filters[1], 4, 2, output_padding=1),
            BasicBlock(filters[1]),
            nn.BatchNorm2d(filters[1]),

            nn.ConvTranspose2d(filters[1], filters[0], 4, 2),
            BasicBlock(filters[0]),
            nn.BatchNorm2d(filters[0]),

            nn.Conv2d(filters[0], 1, 3, 1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, filters):
        'define encoder and decoder'
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(filters)
        self.decoder = Decoder(filters)

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
    filters = [4, 8, 16, 32]
    model = Autoencoder(filters).to(device)
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
