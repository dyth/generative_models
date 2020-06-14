import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import Encoder, Decoder


class BasicBlock(torch.nn.Module):

    def __init__(self, filters=64):
        'residual basic block'
        super().__init__()
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
        super().__init__()
        self.actnorm = torch.nn.Sequential(
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(filters),
        )

    def forward(self, x):
        return self.actnorm(x)


class Res_Encoder(Encoder):

    def __init__(self, filters=[4, 8, 16, 32], bottleneck=10):
        super().__init__()
        self.activate = nn.ELU()
        self.main = nn.Sequential(
            nn.Conv2d(1, filters[0], 3, 1, padding=1),
            self.activate,

            BasicBlock(filters[0]),
            ELU_BatchNorm2d(filters[0]),
            nn.Conv2d(filters[0], filters[1], 5, 2),
            self.activate,

            BasicBlock(filters[1]),
            ELU_BatchNorm2d(filters[1]),
            nn.Conv2d(filters[1], filters[2], 5, 2),
            self.activate,

            BasicBlock(filters[2]),
            ELU_BatchNorm2d(filters[2]),
            nn.Conv2d(filters[2], filters[3], 3, 2),
            self.activate
        )
        self.mean = nn.Conv2d(filters[3], bottleneck, 1, 1)
        self.logvar = nn.Conv2d(filters[3], bottleneck, 1, 1)


class Res_Decoder(Decoder):

    def __init__(self, filters=[4, 8, 16, 32], bottleneck=10):
        super().__init__()
        self.activate = nn.ELU()
        self.main = nn.Sequential(
            nn.Conv2d(bottleneck, filters[-1], 1, 1, bias=False),
            self.activate,

            BasicBlock(filters[-1]),
            ELU_BatchNorm2d(filters[-1]),
            nn.ConvTranspose2d(filters[-1], filters[-2], 3, 2, output_padding=1),
            self.activate,

            BasicBlock(filters[-2]),
            ELU_BatchNorm2d(filters[-2]),
            nn.ConvTranspose2d(filters[-2], filters[-3], 5, 2, output_padding=1),
            self.activate,

            BasicBlock(filters[-3]),
            ELU_BatchNorm2d(filters[-3]),
            nn.ConvTranspose2d(filters[-3], filters[-4], 5, 2, output_padding=1),
            self.activate,

            BasicBlock(filters[-4]),
            ELU_BatchNorm2d(filters[-4]),
            nn.Conv2d(filters[-4], 1, 3, 1, padding=1),
            nn.Sigmoid()
        )
