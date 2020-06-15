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

    def run_one_batch(self, data, optimiser=None, labels=None):
        output, mean, logvar = self(data)
        datasize = data.size(0)
        data = data.reshape(output.shape)
        loss = variational_loss(output, data, mean, logvar) / datasize
        if optimiser is not None:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        return output, loss

    def traverse(self, dataloader, limit=3, steps=10):
        # take the first image from dataloader, get mean, std of latent space
        device = next(self.parameters()).device
        image = dataloader.dataset.__getitem__(0)[0]
        image = image.to(device).unsqueeze(0)
        mean, logvar = self.encoder(image, variational=True)
        width = mean.shape[1]

        # create 10 interpolation points between -3 and 3, multiply by stdev
        interpolation = torch.arange(0, steps, 1).to(device)
        interpolation = limit * (interpolation - (steps-1)/2) / ((steps-1)/2)
        interpolation = torch.cat(width * [interpolation.unsqueeze(0)])
        std = torch.exp(0.5*logvar)[0].view(width, 1)
        interpolation *= std

        # add the interpolations to mean to create the sampling
        mean = torch.cat(width * [torch.cat(steps * [mean]).unsqueeze(0)])
        mean[range(width), :, range(width)] += interpolation
        return self.decoder(mean).view(width * steps, 1, 28, 28), steps
