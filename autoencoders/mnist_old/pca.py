#!/usr/bin/env python
"""
train a convolutional encoder and decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import copy
import numpy as np

from tqdm.autonotebook import tqdm
from torchvision.utils import save_image
from sklearn.decomposition import PCA

from residual import Autoencoder
from dataloaders import *
from fgsm import fgsm_attack

torch.manual_seed(9001)


# def PCA(data, k=2):
#     mean = torch.mean(data, 0)
#     data = data - mean.expand_as(data)
#
#     # svd
#     U, S, V = torch.svd(torch.t(data))
#     return torch.mm(data, U[:, :k])



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


def train_pca(model, pca, train_loader_pca, device, bottleneck):
    for data, _ in train_loader_pca:
        break
    data = data.to(device)
    data = model.encoder(data)
    data = data.view(5000, bottleneck)
    data = data.cpu().detach().numpy()
    space = pca.fit_transform(data)

    percentile5 = np.percentile(space, 2, axis=0)
    percentile95 = np.percentile(space, 98, axis=0)
    return percentile5, percentile95


def traverse_latent_space(model, pca, batch, percentile5, percentile95, device, bottleneck):
    'for every axis in latent space, traverse individually from -1 to 1'
    shift = percentile95 - percentile5
    batch = batch.to(device)
    latent = model.encoder(batch)
    latent = latent.view(bottleneck)
    latent = latent.cpu().numpy()
    latent = pca.transform([latent])[0]
    traversed = []
    for j in range(10):
        traverse = []
        for i in range(bottleneck):
            t = copy.deepcopy(latent)
            t[i] = (j / 9) * shift[i] + percentile5[i]
            t = pca.inverse_transform(t)
            traverse.append(torch.from_numpy(t))
        traverse = torch.stack(traverse)
        traverse = traverse.to(device)
        traverse = traverse.view(bottleneck, bottleneck, 1, 1)
        traverse = model.decoder(traverse)
        traversed.append(traverse)
    return traversed


def test_pca(epoch, model, pca, test_loader_pca, percentile5, percentile95, device, folder, bottleneck):
    'traverse all axes of the latent representation to produce an image'
    # get batch
    for batch, _ in test_loader_pca:
        break
    # traverse latent space and save image
    traversed = traverse_latent_space(model, pca, batch, percentile5, percentile95, device, bottleneck)
    traversed = torch.stack(traversed)
    traversed = traversed.view(10*bottleneck, 1, 28, 28)
    save_image(traversed.cpu(), f'{folder}/{epoch}traverse.png', nrow=bottleneck)


def main():
    batch_size = 64
    test_batch_size = 100
    epochs = 10
    save_model = True
    folder = 'pca3'

    if not os.path.exists(folder):
        os.makedirs(folder)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # filters = [2 ** (i + 2) for i in range(4)]
    filters = [4, 8, 16, 16]
    model = Autoencoder(filters).to(device)
    optimizer = optim.Adam(model.parameters())

    path = 'data'
    train_loader, test_loader = get_mnist(path, use_cuda, batch_size, test_batch_size)
    train_loader_pca, test_loader_pca = get_mnist(path, use_cuda, 5000, 1)

    for epoch in range(1, epochs + 1):
        print(f"\n{epoch}")
        train(model, device, train_loader, optimizer, epoch, folder)
        test(model, device, test_loader, folder, epoch)
        with torch.no_grad():
            bottleneck = filters[-1]
            pca = PCA(n_components=bottleneck)
            percentile5, percentile95 = train_pca(model, pca, train_loader_pca, device, bottleneck)
            test_pca(epoch, model, pca, test_loader_pca, percentile5, percentile95, device, folder, bottleneck)
        if save_model:
            torch.save(model.state_dict(), f"{folder}/{epoch}.pt")



if __name__ == '__main__':
    main()
