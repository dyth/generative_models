# Adapted from https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb
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
from datetime import datetime

from dataloaders import get_mnist


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--train_original", action='store_true')
parser.add_argument("--folder", default=None)
args = parser.parse_args()

# create folder of all arguments
params = vars(args)
default_folder = ''
for p in params:
    if p != 'folder':
        print(f'{p}: {params[p]}')
        default_folder += f'{p}-{params[p]}_'
default_folder += datetime.now().strftime("%y-%m-%d-%H-%M-%S")
args.folder = args.folder if args.folder else default_folder
print('')
print(args.folder)
if not os.path.exists(args.folder):
    os.makedirs(args.folder)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')



class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super().__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


def train_one_batch_efficient(x, G, D, loss, G_opt, D_opt):
    'a more efficient way of training a GAN'
    real_labels = torch.ones(x.size(0), 1).to(device)
    fake_labels = torch.zeros(x.size(0), 1).to(device)
    real_data = x.view(-1, 28 * 28).to(device)
    # sample from N(0, 1)
    z = torch.randn(x.size(0), args.latent_dim).to(device)

    # train the generator
    G.zero_grad()
    fake_data = G(z)
    g_loss = loss(D(fake_data), real_labels)
    g_loss.backward()
    G_opt.step()

    # train the discriminator
    D.zero_grad()
    fake_loss = loss(D(fake_data.detach()), fake_labels)
    real_loss = loss(D(real_data), real_labels)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    D_opt.step()
    return d_loss.data.item(), g_loss.data.item(), fake_data


def train_one_batch_orig(x, G, D, loss, G_opt, D_opt):
    'train one batch of the generator following original GAN paper'
    real_labels = torch.ones(x.size(0), 1).to(device)
    fake_labels = torch.zeros(x.size(0), 1).to(device)
    real_data = x.view(-1, 28 * 28).to(device)

    # train the discriminator
    D.zero_grad()
    d_loss = real_loss + fake_loss
    fake_loss = loss(D(G(z)), fake_labels)
    real_loss = loss(D(real_data), real_labels)
    z = torch.randn(x.size(0), args.latent_dim).to(device)
    d_loss.backward()
    D_opt.step()

    # train the generator
    G.zero_grad()
    z = torch.randn(x.size(0), args.latent_dim).to(device)
    fake_data = G(z)
    g_loss = loss(D(fake_data), real_labels)
    g_loss.backward()
    G_opt.step()
    return d_loss.data.item(), g_loss.data.item(), fake_data


def train_one_epoch(epoch, dataloader, G, D, loss, G_opt, D_opt):
    start = time.time()
    g_losses, d_losses = [], []
    for i, (x, _) in enumerate(dataloader):
        train_one_batch = train_one_batch_efficient
        if args.train_original:
            train_one_batch = train_one_batch_orig
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
    mnist_dim = 28 * 28
    G = Generator(args.latent_dim, mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)
    G_opt = optim.Adam(G.parameters(), lr=args.lr)
    D_opt = optim.Adam(D.parameters(), lr=args.lr)
    loss = nn.BCELoss()
    for epoch in range(1, args.n_epochs+1):
        train_one_epoch(epoch, dataloader, G, D, loss, G_opt, D_opt)



if __name__ == '__main__':
    main()
