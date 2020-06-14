import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoder import Autoencoder


def variational_loss(output, data, mean, logvar):
    'sum reconstruction and divergence losses'
    reconstruction = F.binary_cross_entropy(output, data, reduction='sum')
    divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction + 500 * divergence


class VAE(Autoencoder):

    def __init__(self, encoder, decoder):
        'define encoder and decoder'
        super().__init__(encoder, decoder)

    def repameterise(self, mean, logvar):
        'sample encoding from gaussian of mean and logvar'
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            mean += eps*std
        return mean

    def forward(self, x):
        'pass through encoder and decoder'
        mean, logvar = self.encoder(x, variational=True)
        x = self.repameterise(mean, logvar)
        x = self.decoder(x)
        return x, mean, logvar

    def run_one_batch(self, data, optimiser=None):
        output, mean, logvar = self(data)
        datasize = data.size(0)
        data = data.reshape(output.shape)
        loss = variational_loss(output, data, mean, logvar) / datasize
        if optimiser is not None:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        return output, loss

    def traverse(self, dataloader, limit=3):
        # take the first image from dataloader, get mean std of latent space
        device = next(self.parameters()).device
        image = dataloader.dataset.__getitem__(0)[0]
        image = image.to(device).unsqueeze(0)
        mean, logvar = self.encoder(image, variational=True)
        width = 10#mean.shape[1]

        # # generate -3 to 3 of stdev and multiply by standard deviation
        # step = 2 * limit / width
        # interpolation = torch.arange(-limit, limit, step).unsqueeze(0)
        # interpolations = interpolation.repeat(width, 1).to(device)
        # std = torch.exp(0.5*logvar)
        # std = std.view(width, 1).repeat(1, width)
        # traversals = std * interpolations
        #
        # # substitute traversals into mean, produce images
        # mean = torch.cat(width * [mean.repeat(width, 1).unsqueeze(0)])
        # mean[range(width), :, range(width)] += traversals
        # return self.decoder(mean).view(width**2, 1, 28, 28), width

        # # let's just try to modify the first dimension
        # interpolation = torch.arange(0, width, 1).to(device)
        # interpolation = limit * (interpolation - (width-1)/2) / ((width-1)/2)
        # std = torch.exp(0.5*logvar)[0]
        # interpolation *= std[0]
        # mean = torch.cat(width * [mean])
        # mean[:, 0] += interpolation
        # return self.decoder(mean).view(width, 1, 28, 28), width

        interpolation = torch.arange(0, width, 1).to(device)
        interpolation = limit * (interpolation - (width-1)/2) / ((width-1)/2)
        interpolation = torch.cat(4 * [interpolation.unsqueeze(0)])
        std = torch.exp(0.5*logvar)[0].view(4, 1)
        interpolation *= std
        mean = torch.cat(4 * [torch.cat(width * [mean]).unsqueeze(0)])
        mean[range(4), :, range(4)] += interpolation
        return self.decoder(mean).view(4 * width, 1, 28, 28), width
