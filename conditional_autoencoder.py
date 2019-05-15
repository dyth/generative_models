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


class ConditionalBatchNorm2d(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class BasicBlock(torch.nn.Module):

    def __init__(self, filters=64):
        'residual basic block'
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(filters, filters, 3, 1, padding=1, bias=False)
        self.bn1 = torch.nn.ModuleList([torch.nn.BatchNorm2d(filters, affine=True) for _ in range(10)])
        self.conv2 = torch.nn.Conv2d(filters, filters, 3, 1, padding=1, bias=False)
        self.bn2 = torch.nn.ModuleList([torch.nn.BatchNorm2d(filters, affine=True) for _ in range(10)])

    def conditional_forward(self, i, x):
        out = self.conv1(x)
        out = self.bn1[i](out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2[i](out)
        return x + out

    def forward(self, x):
        return self.conditional_forward(0, x)


class ELU_BatchNorm2d(torch.nn.Module):

    def __init__(self, filters=64):
        super(ELU_BatchNorm2d, self).__init__()
        self.bn = torch.nn.ModuleList([torch.nn.BatchNorm2d(filters, affine=True) for _ in range(10)])

    def conditional_forward(self, i, x):
        x = F.elu(x)
        x = self.bn[i](x)
        return x

    def forward(self, x):
        return self.conditional_forward(0, x)


class ResidualEncoder(torch.nn.Module):

    def __init__(self, filters):
        'define four layers'
        super(ResidualEncoder, self).__init__()
        self.activate = torch.nn.ELU()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, filters[0], 3, 1, padding=1),
            self.activate,

            BasicBlock(filters[0]),
            ELU_BatchNorm2d(filters[0]),
            torch.nn.Conv2d(filters[0], filters[1], 3, 2),
            self.activate,

            BasicBlock(filters[1]),
            ELU_BatchNorm2d(filters[1]),
            torch.nn.Conv2d(filters[1], filters[2], 3, 2),
            self.activate,

            BasicBlock(filters[2]),
            ELU_BatchNorm2d(filters[2]),
            torch.nn.Conv2d(filters[2], filters[3], 3, 2),
            self.activate,

            BasicBlock(filters[3]),
            ELU_BatchNorm2d(filters[3]),
            torch.nn.Conv2d(filters[3], filters[4], 3, 2, bias=False)
        )

    def forward(self, x):
        return self.encoder(x)


class ResidualDecoder(torch.nn.Module):

    def __init__(self, filters):
        'define four layers'
        super(ResidualDecoder, self).__init__()
        self.activate = torch.nn.ELU()

        self.conv1 = torch.nn.ConvTranspose2d(filters[4], filters[3], 3, 2)
        self.eb1 = ELU_BatchNorm2d(filters[3])
        self.block1 = BasicBlock(filters[3])

        self.conv2 = torch.nn.ConvTranspose2d(filters[3], filters[2], 3, 2)
        self.eb2 = ELU_BatchNorm2d(filters[2])
        self.block2 = BasicBlock(filters[2])

        self.conv3 = torch.nn.ConvTranspose2d(filters[2], filters[1], 3, 2)
        self.eb3 = ELU_BatchNorm2d(filters[1])
        self.block3 = BasicBlock(filters[1])

        self.conv4 = torch.nn.ConvTranspose2d(filters[1], filters[0], 3, 2, output_padding=1)
        self.eb4 = ELU_BatchNorm2d(filters[0])
        self.block4 = BasicBlock(filters[0])

        self.conv5 = torch.nn.Conv2d(filters[0], 3, 3, 1, padding=1)

        self.out = torch.nn.Tanh()

    def conditional_forward(self, i, x):
        x = self.conv1(x)
        x = self.eb1.conditional_forward(i, x)
        x = self.block1.conditional_forward(i, x)

        x = self.activate(x)
        x = self.conv2(x)
        x = self.eb2.conditional_forward(i, x)
        x = self.block2.conditional_forward(i, x)

        x = self.activate(x)
        x = self.conv3(x)
        x = self.eb3.conditional_forward(i, x)
        x = self.block3.conditional_forward(i, x)

        x = self.activate(x)
        x = self.conv4(x)
        x = self.eb4.conditional_forward(i, x)
        x = self.block4.conditional_forward(i, x)

        x = self.activate(x)
        x = self.conv5(x)
        x = self.out(x)
        return x

    def forward(self, x):
        return self.conditional_forward(0, x)


class Autoencoder(nn.Module):

    def __init__(self):
        'define encoder and decoder'
        super(Autoencoder, self).__init__()
        filters = [16, 32, 64, 128, 256]
        # filters = [64, 256, 256, 256, 256]
        # filters = [32, 64, 128, 256, 512]
        self.encoder = ResidualEncoder(filters)
        self.decoder = ResidualDecoder(filters)

    def conditional_forward(self, i, x):
        'conditional forward pass through encoder and decoder'
        x = self.encoder(x)
        x = self.decoder.conditional_forward(i, x)
        return x

    def forward(self, x):
        'pass through encoder and decoder'
        return self.conditional_forward(0, x)



def train(model, device, train_loader, optimizer, epoch):
    progress = tqdm(enumerate(train_loader), desc="train", total=len(train_loader))
    model.train()
    train_loss = 0
    for i, (data, labels) in progress:
        data = data.to(device)
        optimizer.zero_grad()
        output = model.conditional_forward(labels, data)
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
        for i, (data, labels) in progress:
            data = data.to(device)
            output = model.conditional_forward(labels, data)
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
    epochs = 10
    save_model = True
    folder = 'conditional'

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
