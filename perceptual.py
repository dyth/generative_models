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


class PerceptualLoss:
    def __init__(self, device):
        with torch.no_grad():
            self.percept = Encoder().to(device)

    def compute(self, output, target):
        output = self.percept(output)
        target = self.percept(target)
        # print(output, target)
        loss = F.mse_loss(output, target)
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
                save_image(data.cpu(), f'{folder}/baseline{epoch}.png', nrow=10)



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
