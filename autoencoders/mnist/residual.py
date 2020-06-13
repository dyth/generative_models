import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import Encoder, Decoder


class BasicBlock(torch.nn.Module):

    def __init__(self, filters=64):
        'residual basic block'
        super(BasicBlock, self).__init__()
        self.residual = torch.nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(filters)
        )

    def forward(self, x):
        return x + self.residual(x)


class ELU_BatchNorm2d(torch.nn.Module):

    def __init__(self, filters=64):
        super(ELU_BatchNorm2d, self).__init__()
        self.actnorm = torch.nn.Sequential(
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(filters)
        )

    def forward(self, x):
        return self.actnorm(x)


class Res_Encoder(Encoder):

    def __init__(self, filters=[4, 8, 16, 32]):
        'define four layers'
        super(Res_Encoder, self).__init__()
        self.activate = nn.ELU()
        self.main = nn.Sequential(
            nn.Conv2d(1, filters[0], 3, 1, padding=1),
            self.activate,

            BasicBlock(filters[0]),
            self.activate,
            nn.BatchNorm2d(filters[0]),
            nn.Conv2d(filters[0], filters[1], 5, 2),
            self.activate,

            BasicBlock(filters[1]),
            self.activate,
            nn.BatchNorm2d(filters[1]),
            nn.Conv2d(filters[1], filters[2], 5, 2),
            self.activate,

            BasicBlock(filters[2]),
            self.activate,
            nn.BatchNorm2d(filters[2])
        )
        self.mean = nn.Conv2d(filters[2], filters[3], 3, 2)
        self.logvar = nn.Conv2d(filters[2], filters[3], 3, 2)


class Res_Decoder(Decoder):

    def __init__(self, filters=[4, 8, 16, 32]):
        'define four layers'
        super(Res_Decoder, self).__init__()
        self.activate = nn.ELU()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(filters[-1], filters[-2], 3, 2, output_padding=1),
            self.activate,
            nn.BatchNorm2d(filters[-2]),
            BasicBlock(filters[-2]),

            self.activate,
            nn.ConvTranspose2d(filters[-2], filters[-3], 5, 2, output_padding=1),
            self.activate,
            nn.BatchNorm2d(filters[-3]),
            BasicBlock(filters[-3]),

            self.activate,
            nn.ConvTranspose2d(filters[-3], filters[-4], 5, 2, output_padding=1),
            self.activate,
            nn.BatchNorm2d(filters[-4]),
            BasicBlock(filters[-4]),

            self.activate,
            nn.Conv2d(filters[-4], 1, 3, 1, padding=1),
            nn.Sigmoid()
        )
