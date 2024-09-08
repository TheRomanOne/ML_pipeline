import torch
import torch.nn as nn
from custom_models.SuperRsolution import SuperResolution
from custom_models.VAE import VAE

class VAE_SR(VAE):
    def __init__(self, params, input_shape):
        super(VAE_SR, self).__init__(params, input_shape)

        self.sr_1 = SuperResolution()
        self.sr_2 = SuperResolution()
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def from_latent(self, x):
      x = self.relu(self.decoder(x))
      x = self.sr_2(self.sr_1(x))
      return x