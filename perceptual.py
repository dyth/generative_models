#!/usr/bin/env python
"""
train an encoder with perceptual loss
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
from convolutional import Autoencoder, Encoder

torch.manual_seed(9001)


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        'define four layers'
        super(PerceptualLoss, self).__init__()
        with torch.no_grad():
            self.conv1 = nn.Conv2d(1, 8, 5, 2)
            self.conv2 = nn.Conv2d(8, 16, 5, 2)
            self.fc1 = nn.Linear(256, 96)
            self.fc2 = nn.Linear(96, 32)
            self.to(device)

    def compute(self, output, target):
        'compare activations at each layer'
        output1 = F.relu(self.conv1(output))
        output2 = F.relu(self.conv2(output1))
        output2 = output2.view(-1, 256)
        output3 = F.relu(self.fc1(output2))
        output4 = F.relu(self.fc2(output3))

        target1 = F.relu(self.conv1(target))
        target2 = F.relu(self.conv2(target1))
        target2 = target2.view(-1, 256)
        target3 = F.relu(self.fc1(target2))
        target4 = F.relu(self.fc2(target3))

        loss = F.mse_loss(output, target)
        loss += F.mse_loss(output1, target1)
        loss += F.mse_loss(output2, target2)
        loss += F.mse_loss(output3, target3)
        loss += F.mse_loss(output4, target4)
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
                output = output.view(100, 1, 28, 28)
                data = data.view(100, 1, 28, 28)
                save_image(output.cpu(), f'{folder}/{epoch}.png', nrow=10)
                save_image(data.cpu(), f'{folder}/{epoch}baseline.png', nrow=10)



def main():
    batch_size = 64
    test_batch_size = 100
    epochs = 10
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
    train_loader, test_loader = get_mnist(path, use_cuda, batch_size, test_batch_size)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss)
        test(model, device, test_loader, folder, epoch, loss)
        print("")
        if save_model:
            torch.save(model.state_dict(), f"{folder}/{epoch}.pt")



if __name__ == '__main__':
    main()
