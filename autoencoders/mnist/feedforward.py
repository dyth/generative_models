import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import Encoder, Decoder

class Flatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input, 1)


class FNN_Encoder(Encoder):

    def __init__(self):
        super(FNN_Encoder, self).__init__()
        self.main = nn.Sequential(
            Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mean = nn.Linear(128, 64)
        self.logvar = nn.Linear(128, 64)


class FNN_Decoder(Decoder):

    def __init__(self):
        super(FNN_Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
