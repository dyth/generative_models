#!/usr/bin/env python
"""
train an encoder and a decoder
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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 16, 4, 4)
        x = F.relu(self.conv2(x))
        x = F.sigmoid(self.conv1(x))
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
    progress = tqdm(enumerate(train_loader), desc="", total=len(train_loader))
    model.train()
    total_loss = 0
    for i, (data, _) in progress:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, data)
        loss.backward()
        optimizer.step()
        log_interval = 10
        total_loss += loss
        progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))


def test(model, device, test_loader, folder, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, data, reduction='sum').item()
            test_loss /= len(test_loader.dataset)
            if i == 0:
                output = output.view(100, 1, 28, 28)
                data = data.view(100, 1, 28, 28)
                save_image(output.cpu(), f'{folder}/{epoch}.png', nrow=10)
                save_image(data.cpu(), f'{folder}/baseline{epoch}.png', nrow=10)
        print(f'\nTest set: Average loss: {test_loss:.4f}\n')



def main():
    batch_size = 64
    test_batch_size = 100
    epochs = 10
    save_model = True
    folder = 'convolutional'

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
        if save_model:
            torch.save(model.state_dict(), f"{folder}/{epoch}.pt")



if __name__ == '__main__':
    main()
