import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def forward(self, x, variational=False):
        middle = self.main(x)
        mean = self.mean(middle)
        if variational:
            logvar = self.logvar(middle)
            return mean, logvar
        return mean


class Decoder(nn.Module):

    def forward(self, x):
        return self.main(x)


class Autoencoder(nn.Module):

    def __init__(self, encoder, decoder):
        'define encoder and decoder'
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        'pass through encoder and decoder'
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def run_one_batch(self, data, optimiser=None):
        output = self(data)
        data = data.reshape(output.shape)
        loss = F.binary_cross_entropy(output, data)
        if optimiser is not None:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        return output, loss
