import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from tqdm.autonotebook import tqdm
from torchvision.utils import save_image

from models import models, losses


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', default='fnn',
                    help='what model architecture to use')
parser.add_argument('--loss', default='ae',
                    help='what loss function to train architecture')
parser.add_argument('--traverse', action='store_false', default=True,
                    help='produce and image showing latent traversals')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='save the current model')
parser.add_argument('--save-image', action='store_false', default=True,
                    help='save autoencoder images')
parser.add_argument('--no-tqdm', action='store_true', default=False,
                    help='use tqdm')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
random.seed(args.seed)

folder = f'images/{args.loss}_{args.model}'
if not os.path.exists(folder):
    os.makedirs(folder)


def run_one_epoch(model, dataloader, name, epoch, optimiser=None):
    'train if optimiser is present; save 64 images; print loss'
    if optimiser is not None:
        model.train()
    else:
        model.eval()
    device = next(model.parameters()).device

    with torch.set_grad_enabled(optimiser is not None):
        total_loss = 0
        progress = enumerate(dataloader)
        if not args.no_tqdm:
            progress = tqdm(progress, total=len(dataloader))
        for i, (data, labels) in progress:
            data, labels = data.to(device), labels.to(device)
            output, loss = model.run_one_batch(data, optimiser=optimiser, labels=labels)
            total_loss += loss
            if not args.no_tqdm:
                progress.set_description(f"{name} loss: {total_loss/(i+1):.4f}")
            if i == 0 and args.save_image and optimiser is None:
                data = data[:64, ].cpu().view(64, 1, 28, 28)
                output = output[:64, ].cpu().view(64, 1, 28, 28)
                save = {'nrow': 8, 'pad_value': 64}
                save_image(data, f'{folder}/{epoch}baseline.png', **save)
                save_image(output, f'{folder}/{epoch}.png', **save)

    if args.no_tqdm:
        print(f'{name}: Average loss: {total_loss/(i+1) :.4f}')


def get_data():
    path = '../../data'
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=True, transform=t),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False, download=True, transform=t),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def main():
    train_loader, test_loader = get_data()
    encoder = models[args.model][0]()
    decoder = models[args.model][1]()
    model = losses[args.loss](encoder, decoder).to(device)

    args.traverse = args.traverse and (args.loss != 'ae')

    optimiser = optim.Adam(model.parameters())
    for epoch in range(1, args.epochs + 1):
        print(f'\n{epoch}')
        run_one_epoch(model, train_loader, 'train', epoch, optimiser=optimiser)
        run_one_epoch(model, test_loader, 'test', epoch)
        if args.traverse:
            output, width = model.traverse(test_loader)
            save_image(output.cpu(), f'{folder}/{epoch}traverse.png',
                nrow=width, pad_value=64)

    if args.save_model:
        torch.save(model.state_dict(), f"{folder}/{epoch}.pt")


if __name__ == '__main__':
    main()
