#!/usr/bin/env python
"""
download mnist
"""
import torch.utils.data
from torchvision import datasets, transforms


def get_mnist(path, use_cuda, batch_size):
    'download into folder data if folder does not exist, then create dataloader'
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = datasets.MNIST(path, train=True, download=True, transform=t)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, **kwargs
    )


def get_mnist(path, use_cuda, batch_size):
    'download into folder data if folder does not exist, then create dataloader'
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    t = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = datasets.MNIST(path, train=True, download=True, transform=t)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
