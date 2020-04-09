# Model from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import os
import argparse
import time

from dataloaders import get_mnist


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--folder", default='dcgan')
args = parser.parse_args()
print(args)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

if not os.path.exists(args.folder):
    os.makedirs(args.folder)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 28 // 4
        self.l1 = nn.Sequential(nn.Linear(args.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        return self.conv_blocks(out)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 2
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        return self.adv_layer(out)


def train_one_batch(x, G, D, loss, G_opt, D_opt):
    real_labels = torch.ones(x.size(0), 1).to(device)
    fake_labels = torch.zeros(x.size(0), 1).to(device)
    real_data = x.to(device)
    # sample from N(0, 1)
    z = torch.randn(x.size(0), args.latent_dim).to(device)

    # train the generator
    G.zero_grad()
    fake_data = G(z)
    f = D(fake_data)
    g_loss = loss(f, real_labels)
    g_loss.backward()
    G_opt.step()

    # train the discriminator
    D.zero_grad()
    # D_fake_loss = loss(D(G(z)), fake_labels)
    fake_loss = loss(D(fake_data.detach()), fake_labels)
    real_loss = loss(D(real_data), real_labels)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    D_opt.step()
    return d_loss.data.item(), g_loss.data.item(), fake_data


def train_one_epoch(epoch, dataloader, G, D, loss, G_opt, D_opt):
    start = time.time()
    g_losses, d_losses = [], []
    for i, (x, _) in enumerate(dataloader):
        g_loss, d_loss, fake_data = train_one_batch(x, G, D, loss, G_opt, D_opt)
        g_losses.append(g_loss)
        d_losses.append(d_loss)
        if i == 0:
            name = f'{args.folder}/{epoch}.png'
            save_image(fake_data.view(fake_data.size(0), 1, 28, 28), name)
    end = time.time()
    print(
        f'[{epoch}/{args.n_epochs}] '\
        f'{end - start:.3f}s: '\
        f'D loss {torch.mean(torch.FloatTensor(d_losses)):.3f}, '\
        f'G loss {torch.mean(torch.FloatTensor(g_losses)):.3f}'
    )


def main():
    dataloader = get_mnist('../data', use_cuda, args.batch_size)
    G = Generator().to(device)
    D = Discriminator().to(device)
    G_opt = optim.Adam(G.parameters(), lr=args.lr)
    D_opt = optim.Adam(D.parameters(), lr=args.lr)
    loss = nn.BCELoss()
    for epoch in range(1, args.n_epochs+1):
        train_one_epoch(epoch, dataloader, G, D, loss, G_opt, D_opt)



if __name__ == '__main__':
    main()
