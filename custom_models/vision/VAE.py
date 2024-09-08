import torch, yaml
import torch.nn as nn
import numpy as np
from utils.architecture_parser import parse_architecture




class Encoder(nn.Module):
    def __init__(self, latent_dim, input_shape, architecture):
        super(Encoder, self).__init__()        
        seq, layer_shapes = parse_architecture(architecture, input_shape)
        self.convolutions = nn.Sequential(*seq)

        self.flattened_size = layer_shapes[-1][0] * layer_shapes[-1][1] * architecture[-2]['params']['out_channels']
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_shape, init_size, architecture):
        super(Decoder, self).__init__()
        self.input_shape = np.array(input_shape) / 2
        self.input_shape = np.array([int(self.input_shape[0]), int(self.input_shape[1])])
        self.fc = nn.Linear(latent_dim, init_size * self.input_shape[0] * self.input_shape[1])

        seq, layer_shapes = parse_architecture(architecture, self.input_shape)
        self.seq = seq
        self.layer_shapes = layer_shapes
        self.deconvolutions = nn.Sequential(*seq)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), -1, self.input_shape[0], self.input_shape[1])

        z = self.deconvolutions(z)
        return z

class VAE(nn.Module):
    def __init__(self, params, input_shape):
        super(VAE, self).__init__()

        with open('custom_models/sequence_config/encoder.yaml', 'r') as yaml_file:
            encoder_architecture = yaml.safe_load(yaml_file)

        with open('custom_models/sequence_config/decoder.yaml', 'r') as yaml_file:
            decoder_architecture = yaml.safe_load(yaml_file)

        self.encoder = Encoder(latent_dim=params['latent_dim'], input_shape=input_shape, architecture=encoder_architecture)
        self.decoder = Decoder(latent_dim=params['latent_dim'], input_shape=input_shape, init_size=params['max_size'], architecture=decoder_architecture)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent = self.reparameterize(mu, logvar)
        recon_x = self.from_latent(latent)

        return recon_x, mu, logvar

    def from_latent(self, x):
      return self.decoder(x)
      