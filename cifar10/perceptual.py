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

from residual import Autoencoder, BasicBlock, ELU_BatchNorm2d
from dataloaders import *

torch.manual_seed(9001)


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        'define four layers'
        super(PerceptualLoss, self).__init__()
        with torch.no_grad():
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
                torch.nn.Conv2d(128, 256, 3, 2, bias=False)
            )
            self.to(device)

    def forward_list(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        x = [x, x1, x2, x3, x4, x5]
        return x

    def compute(self, output, target):
        'compare activations at each layer'
        output = self.forward_list(output)
        target = self.forward_list(target)
        loss = sum([F.mse_loss(o, t) for o, t in zip(output, target)])
        return loss


def train(model, device, train_loader, optimizer, epoch, loss):
    progress = tqdm(enumerate(train_loader), desc="train", total=len(train_loader))
    model.train()
    train_loss = 0
    for i, (data, _) in progress:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        batch_loss = loss.compute(output, data)
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss
        progress.set_description("train loss: {:.4f}".format(train_loss/(i+1)))


def test(model, device, test_loader, folder, epoch, loss):
    progress = tqdm(enumerate(test_loader), desc="test", total=len(test_loader))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in progress:
            data = data.to(device)
            output = model(data)
            test_loss += loss.compute(output, data)
            progress.set_description("test loss: {:.4f}".format(test_loss/(i+1)))
            if i == 0:
                output = output.view(100, 3, 32, 32)
                data = data.view(100, 3, 32, 32)
                save_image(output.cpu(), f'{folder}/{epoch}.png', nrow=10)
                save_image(data.cpu(), f'{folder}/{epoch}baseline.png', nrow=10)



def main():
    batch_size = 64
    test_batch_size = 100
    epochs = 20
    save_model = True
    folder = 'perceptual'

    if not os.path.exists(folder):
        os.makedirs(folder)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters())
    loss = PerceptualLoss(device)

    path = 'data'
    train_loader, test_loader = get_cifar10(path, use_cuda, batch_size, test_batch_size)

    for epoch in range(1, epochs + 1):
        print(f"\n{epoch}")
        train(model, device, train_loader, optimizer, epoch, loss)
        test(model, device, test_loader, folder, epoch, loss)
        if save_model:
            torch.save(model.state_dict(), f"{folder}/{epoch}.pt")



if __name__ == '__main__':
    main()
