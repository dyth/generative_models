from feedforward import FNN_Encoder, FNN_Decoder
from cnn import CNN_Encoder, CNN_Decoder
from residual import Res_Encoder, Res_Decoder

from autoencoder import Autoencoder
from variational import VAE

models = {
    'fnn': [FNN_Encoder, FNN_Decoder],
    'cnn': [CNN_Encoder, CNN_Decoder],
    'res': [Res_Encoder, Res_Decoder]
}

losses = {
    'ae': Autoencoder,
    'vae': VAE
}
