import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoder import Autoencoder


def variational_loss(output, data, mean, logvar):
    'sum reconstruction and divergence losses'
    reconstruction = F.binary_cross_entropy(output, data)
    divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction + divergence


class VAE(Autoencoder):

    def __init__(self, encoder, decoder):
        'define encoder and decoder'
        super(VAE, self).__init__(encoder, decoder)

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
        data = data.reshape(output.shape)
        loss = variational_loss(output, data, mean, logvar)
        if optimiser is not None:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        return output, loss
