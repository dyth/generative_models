import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import Encoder, Decoder

bottleneck = 4


class Flatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input, 1)


class FNN_Encoder(Encoder):

    def __init__(self):
        super().__init__()
        self.activate = nn.ELU()
        self.main = nn.Sequential(
            Flatten(),
            nn.Linear(784, 64),
            self.activate
        )
        self.mean = nn.Linear(64, bottleneck)
        self.logvar = nn.Linear(64, bottleneck)


class FNN_Decoder(Decoder):

    def __init__(self):
        super().__init__()
        self.activate = nn.ELU()
        self.main = nn.Sequential(
            nn.Linear(bottleneck, 64, bias=False),
            self.activate,
            nn.Linear(64, 784),
            nn.Sigmoid()
        )
