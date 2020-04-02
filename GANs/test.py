# Adapted from https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import os

from dataloaders import get_mnist

batch_size = 100
z_dim = 100
lr = 0.0001
n_epoch = 10
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

folder = 'gan'
if not os.path.exists(folder):
    os.makedirs(folder)


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


def train_one_batch(x, G, D, loss, G_opt, D_opt):
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    real_data = x.view(-1, 28 * 28).to(device)
    # sample from N(0, 1)
    z = torch.randn(batch_size, z_dim).to(device)

    # train the generator
    G.zero_grad()
    fake_data = G(z)
    g_loss = loss(D(fake_data), real_labels)
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


def train_one_epoch(epoch, n_epoch, dataloader, G, D, loss, G_opt, D_opt):
    g_losses, d_losses = [], []
    for i, (x, _) in enumerate(dataloader):
        g_loss, d_loss, fake_data = train_one_batch(x, G, D, loss, G_opt, D_opt)
        g_losses.append(g_loss)
        d_losses.append(d_loss)
        if i == 0:
            name = f'{folder}/{epoch}.png'
            save_image(fake_data.view(fake_data.size(0), 1, 28, 28), name)
    print('[%d/%d]: D loss: %.3f, G loss: %.3f' % (
        epoch, n_epoch,
        torch.mean(torch.FloatTensor(d_losses)),
        torch.mean(torch.FloatTensor(g_losses)))
    )


def main():
    dataloader = get_mnist('../data', use_cuda, batch_size)
    mnist_dim = 28 * 28
    G = Generator(z_dim, mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)
    G_opt = optim.Adam(G.parameters(), lr=lr)
    D_opt = optim.Adam(D.parameters(), lr=lr)
    loss = nn.BCELoss()
    for epoch in range(1, n_epoch+1):
        train_one_epoch(epoch, n_epoch, dataloader, G, D, loss, G_opt, D_opt)



if __name__ == '__main__':
    main()
