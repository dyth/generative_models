import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import Encoder, Decoder


class CNN_Encoder(Encoder):

    def __init__(self, filters=[4, 8, 16, 32]):
        super(CNN_Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, filters[0], 3, 1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[1], 5, 2),
            nn.ReLU(),

            nn.Conv2d(filters[1], filters[1], 3, 1, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[2], 5, 2),
            nn.ReLU(),

            nn.Conv2d(filters[2], filters[2], 3, 1, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU()
        )
        self.mean = nn.Conv2d(filters[2], filters[3], 3, 2, bias=False)
        self.logvar = nn.Conv2d(filters[2], filters[3], 3, 2, bias=False)


class CNN_Decoder(Decoder):

    def __init__(self, filters=[4, 8, 16, 32]):
        super(CNN_Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(filters[-1], filters[-2], 3, 2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters[-2]),
            nn.Conv2d(filters[-2], filters[-2], 3, 1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(filters[-2], filters[-3], 5, 2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters[-3]),
            nn.Conv2d(filters[-3], filters[-3], 3, 1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(filters[-3], filters[-4], 5, 2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters[-4]),
            nn.Conv2d(filters[-4], 1, 3, 1, padding=1),
            nn.Sigmoid()
        )
