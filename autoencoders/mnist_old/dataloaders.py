#!/usr/bin/env python
"""
download mnist
"""
import torch.utils.data
from torchvision import datasets, transforms


def get_mnist(path, use_cuda, batch_size, test_batch_size):
    'download into folder data if folder does not exist, then create dataloader'
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=True, transform=t),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False, download=True, transform=t),
        batch_size=test_batch_size, shuffle=True, **kwargs
    )
    return train_loader, test_loader


def get_2d_mnist(path, use_cuda, batch_size, test_batch_size):
    'download into folder data if folder does not exist, then create dataloader'

    t = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=True, transform=t),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False, download=True, transform=t),
        batch_size=test_batch_size, shuffle=True, **kwargs
    )
    return train_loader, test_loader


def get_cifar10(path, use_cuda, batch_size, test_batch_size):
    'download into folder data if folder does not exist, then create dataloader'
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(path, train=True, download=True, transform=t),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(path, train=False, download=True, transform=t),
        batch_size=test_batch_size, shuffle=True, **kwargs
    )
    return train_loader, test_loader


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    path = '../data'
    get_mnist(path, use_cuda, 64, 1000)
    get_cifar10(path, use_cuda, 64, 1000)
