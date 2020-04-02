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

from residual import BasicBlock, ELU_BatchNorm2d
from dataloaders import *

torch.manual_seed(9001)


class PerceptualEncoder(nn.Module):
    def __init__(self):
        'define four layers'
        super(PerceptualEncoder, self).__init__()
        self.activate = torch.nn.ELU()
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 1, padding=1),
            self.activate
        )
        self.encoder2 = torch.nn.Sequential(
            BasicBlock(16),
            ELU_BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 3, 2),
            self.activate
        )
        self.encoder3 = torch.nn.Sequential(
            BasicBlock(32),
            ELU_BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 3, 2),
            self.activate
        )
        self.encoder4 = torch.nn.Sequential(
            BasicBlock(64),
            ELU_BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 3, 2),
            self.activate
        )
        self.encoder5 = torch.nn.Sequential(
            BasicBlock(128),
            ELU_BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 3, 2)
        )
        self.encoder6 = torch.nn.Conv2d(256, 256, 1, 1)

    def forward_list(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        pc = self.encoder6(x5)
        x = [x, x5]
        return x, pc


class PerceptualDecoder(torch.nn.Module):

    def __init__(self):
        'define four layers'
        super(PerceptualDecoder, self).__init__()
        self.activate = torch.nn.ELU()
        self.decoder0 = torch.nn.Conv2d(256, 256, 1, 1)
        self.decoder1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 3, 2),
            ELU_BatchNorm2d(128),
            BasicBlock(128)
        )
        self.decoder2 = torch.nn.Sequential(
            self.activate,
            torch.nn.ConvTranspose2d(128, 64, 3, 2),
            ELU_BatchNorm2d(64),
            BasicBlock(64)
        )
        self.decoder3 = torch.nn.Sequential(
            self.activate,
            torch.nn.ConvTranspose2d(64, 32, 3, 2),
            ELU_BatchNorm2d(32),
            BasicBlock(32)
        )
        self.decoder4 = torch.nn.Sequential(
            self.activate,
            torch.nn.ConvTranspose2d(32, 16, 3, 2, output_padding=1),
            ELU_BatchNorm2d(16),
            BasicBlock(16)
        )
        self.decoder5 = torch.nn.Sequential(
            self.activate,
            torch.nn.Conv2d(16, 3, 3, 1, padding=1),
            torch.nn.Tanh()
        )

    def forward_list(self, pc):
        x = self.decoder0(pc)
        x1 = self.decoder1(x)
        x2 = self.decoder2(x1)
        x3 = self.decoder3(x2)
        x4 = self.decoder4(x3)
        x5 = self.decoder5(x4)
        x = [x5, x]
        return x


class Autoencoder(nn.Module):

    def __init__(self):
        'define encoder and decoder'
        super(Autoencoder, self).__init__()
        self.encoder = PerceptualEncoder()
        self.decoder = PerceptualDecoder()

    def forward(self, x):
        'pass through encoder and decoder'
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def compute(output, target):
    'compare activations at each layer'
    loss = sum([F.mse_loss(o, t) for o, t in zip(output, target)])
    return loss


def train(model, device, train_loader, optimizer, epoch):
    progress = tqdm(enumerate(train_loader), desc="train", total=len(train_loader))
    model.train()
    train_loss = 0
    for i, (data, _) in progress:
        data = data.to(device)
        optimizer.zero_grad()

        features_in, hidden = model.encoder.forward_list(data)
        features_out = model.decoder.forward_list(hidden)
        output = features_out[0]
        batch_loss = compute(features_in, features_out)

        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss
        progress.set_description("train loss: {:.4f}".format(train_loss/(i+1)))


def test(model, device, test_loader, folder, epoch):
    progress = tqdm(enumerate(test_loader), desc="test", total=len(test_loader))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in progress:
            data = data.to(device)

            features_in, hidden = model.encoder.forward_list(data)
            features_out = model.decoder.forward_list(hidden)
            output = features_out[0]

            test_loss += compute(features_in, features_out)
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
    folder = 'pcautoencoder'

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
