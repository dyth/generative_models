import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import Encoder, Decoder


class CNN_Encoder(Encoder):

    def __init__(self, filters=[4, 8, 16, 32], bottleneck=10):
        super().__init__()
        self.activate = nn.ELU()
        self.main = nn.Sequential(
            nn.Conv2d(1, filters[0], 3, 1, padding=1),
            nn.BatchNorm2d(filters[0]),
            self.activate,
            nn.Conv2d(filters[0], filters[1], 5, 2),
            self.activate,

            nn.Conv2d(filters[1], filters[1], 3, 1, padding=1),
            nn.BatchNorm2d(filters[1]),
            self.activate,
            nn.Conv2d(filters[1], filters[2], 5, 2),
            self.activate,

            nn.Conv2d(filters[2], filters[2], 3, 1, padding=1),
            nn.BatchNorm2d(filters[2]),
            self.activate,
            nn.Conv2d(filters[2], filters[3], 3, 2),
            self.activate,
        )
        self.mean = nn.Conv2d(filters[3], bottleneck, 1, 1)
        self.logvar = nn.Conv2d(filters[3], bottleneck, 1, 1)


class CNN_Decoder(Decoder):

    def __init__(self, filters=[4, 8, 16, 32], bottleneck=10):
        super().__init__()
        self.activate = nn.ELU()
        self.main = nn.Sequential(
            nn.Conv2d(bottleneck, filters[-1], 1, 1),
            self.activate,

            nn.ConvTranspose2d(filters[-1], filters[-2], 3, 2, output_padding=1),
            self.activate,
            nn.BatchNorm2d(filters[-2]),
            nn.Conv2d(filters[-2], filters[-2], 3, 1, padding=1),
            self.activate,

            nn.ConvTranspose2d(filters[-2], filters[-3], 5, 2, output_padding=1),
            self.activate,
            nn.BatchNorm2d(filters[-3]),
            nn.Conv2d(filters[-3], filters[-3], 3, 1, padding=1),
            self.activate,

            nn.ConvTranspose2d(filters[-3], filters[-4], 5, 2, output_padding=1),
            self.activate,
            nn.BatchNorm2d(filters[-4]),
            nn.Conv2d(filters[-4], 1, 3, 1, padding=1),
            nn.Sigmoid()
        )
