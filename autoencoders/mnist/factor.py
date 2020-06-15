import torch
import torch.nn as nn
import torch.nn.functional as F
from variational import VAE, variational_loss
# this is really good
# https://github.com/1Konny/FactorVAE/blob/master/solver.py

def disc_one_batch(model, data, target):
    logits = model(data)
    output = F.log_softmax(logits, dim=1)
    loss = F.nll_loss(output, target)
    pred = output.argmax(dim=1, keepdim=True)
    return pred, loss


class Discrimator(nn.Module):

    def __init__(self):
        super().__init__()
        self.fnn = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        return self.fnn(x)


class Factor_VAE(VAE):

    def __init__(self, encoder, decoder):
        'define encoder and decoder'
        super().__init__(encoder, decoder)
        self.disc = Discrimator()

    def run_one_batch(self, data, optimiser=None, labels=None):
        output, mean, logvar = self(data)
        mean2, logvar2 = mean.clone(), logvar.clone()
        datasize = data.size(0)
        data = data.reshape(output.shape)
        loss = variational_loss(output, data, mean, logvar) / datasize

        middle = self.repameterise(mean2, logvar2)
        pred, disc_loss = disc_one_batch(self.disc, middle, labels)
        loss += disc_loss

        if optimiser is not None:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        return output, loss
