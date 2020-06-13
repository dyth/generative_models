#!/usr/bin/env python
"""
train an encoder with fgsm attacks
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

torch.manual_seed(9001)


class Encoder(nn.Module):

    def __init__(self):
        'define four layers'
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, 2)
        self.fc1 = nn.Linear(256, 96)
        self.fc2 = nn.Linear(96, 32)

    def forward(self, x):
        'convolution'
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, 256)
        x = F.elu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class Decoder(nn.Module):

    def __init__(self):
        'define four layers'
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(32, 96)
        self.fc2 = nn.Linear(96, 256)
        self.conv2 = nn.ConvTranspose2d(16, 8, 5, 2, output_padding=1)
        self.conv1 = nn.ConvTranspose2d(8, 1, 5, 2, output_padding=1)

    def forward(self, x):
        'deconvolution'
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = x.view(-1, 16, 4, 4)
        x = F.elu(self.conv2(x))
        x = torch.tanh(self.conv1(x))
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



# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    # Return the perturbed image
    return perturbed_image


def train(model, device, train_loader, optimizer, epoch, folder):
    progress = tqdm(enumerate(train_loader), desc="train", total=len(train_loader))
    model.train()
    train_loss = 0
    grad = None
    for i, (data, _) in progress:
        data = data.to(device)
        optimizer.zero_grad()

        hidden = model.encoder(data)
        hidden.retain_grad()
        output = model.decoder(hidden)

        batch_loss = F.mse_loss(output, data)
        batch_loss.backward()
        optimizer.step()

        perturbed = model.encoder(data)
        perturbed = fgsm_attack(perturbed, 0.5, hidden.grad.data)
        output2 = model.decoder(perturbed)

        loss = F.mse_loss(output2, data)
        loss.backward()
        optimizer.step()

        if i == 0:
            output = output.view(64, 1, 28, 28)
            output2 = output2.view(64, 1, 28, 28)
            save_image(output2.cpu(), f'{folder}/{epoch}.png', nrow=8)
            save_image(output.cpu(), f'{folder}/{epoch}baseline.png', nrow=8)

        train_loss += batch_loss
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
    epochs = 100
    save_model = True

    folder = 'fgsm'
    if not os.path.exists(folder):
        os.makedirs(folder)

    folder2 = 'fgsm/perturbation'
    if not os.path.exists(folder2):
        os.makedirs(folder2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters())

    path = 'data'
    train_loader, test_loader = get_mnist(path, use_cuda, batch_size, test_batch_size)

    for epoch in range(1, epochs + 1):
        print(epoch)
        train(model, device, train_loader, optimizer, epoch, folder2)
        test(model, device, test_loader, folder, epoch)
        print("")
        if save_model:
            torch.save(model.state_dict(), f"{folder}/{epoch}.pt")



if __name__ == '__main__':
    main()
