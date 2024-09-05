import torch
import torch.nn as nn
from models.SuperRsolution import SuperResolution
from utils.utils import conv2d_output_shape, deconv2d_output_shape
import numpy as np

decoder_architecture = [
# ______________   LAYER 1  ______________
            {
                'function': 'deconv2d',
                'params': {
                    'in_channels': -1, # take size from config
                    'out_channels': 64,
                    'kernel_size': 4,
                    'stride': 2,
                    'padding': 1,
                }
            },
            {
                'function': 'batchNorm2d',
                'params': {
                    'size': 64,
                }
            },
            {
                'function': 'relu',
                'params': {'negative_slope': 0.2}
            },

# ______________   LAYER 2  ______________
            {
                'function': 'deconv2d',
                'params': {
                    'in_channels': 64,
                    'out_channels': 32,
                    'kernel_size': 2,
                    'stride': 1,
                    'padding': 0,
                }
            },
            {
                'function': 'batchNorm2d',
                'params': {
                    'size': 32,
                }
            },
            {
                'function': 'relu',
                'params': {'negative_slope': 0.2}
            },

# ______________   LAYER 3  ______________
            {
                'function': 'conv2d',
                'params': {
                    'in_channels': 32,
                    'out_channels': 3,
                    'kernel_size': 2,
                    'stride': 1,
                    'padding': 0,
                }
            }
        ]
  

encoder_architecture = [
# ______________   LAYER 1  ______________
            {
                'function': 'conv2d',
                'params': {
                    'in_channels': 3,
                    'out_channels': 32,
                    'kernel_size': 4,
                    'stride': 2,
                    'padding': 1,
                }
            },
            {
                'function': 'batchNorm2d',
                'params': {
                    'size': 32,
                }
            },
            {
                'function': 'relu',
                'params': {'negative_slope': 0.2}
            },

# ______________   LAYER 2  ______________
            {
                'function': 'conv2d',
                'params': {
                    'in_channels': 32,
                    'out_channels': 64,
                    'kernel_size': 4,
                    'stride': 2,
                    'padding': 1,
                }
            },
            {
                'function': 'batchNorm2d',
                'params': {
                    'size': 64,
                }
            },
            {
                'function': 'relu',
                'params': {'negative_slope': 0.2}
            },

# ______________   LAYER 3  ______________
            {
                'function': 'conv2d',
                'params': {
                    'in_channels': 64,
                    'out_channels': 128,
                    'kernel_size': 4,
                    'stride': 2,
                    'padding': 1,
                }
            }
        ]



def parse_architecture(input_shape, architecture):
    seq = []
    layer_shapes = [input_shape]
    for a in architecture:
        a_params = a['params']
        if a['function'] == 'conv2d':
            # Add functional layer
            action = nn.Conv2d(
                in_channels=a_params['in_channels'],
                out_channels=a_params['out_channels'],
                kernel_size=a_params['kernel_size'],
                stride=a_params['stride'],
                padding=a_params['padding']
            )

            # keep track of the convolution shapes
            layer_shapes.append(
                conv2d_output_shape(
                    input_shape=layer_shapes[-1],
                    kernel_size=a_params['kernel_size'],
                    stride=a_params['stride'],
                    padding=a_params['padding']
                )
            )
        if a['function'] == 'deconv2d':
            # Add functional layer

            # TODO: remove hack
            in_channels = a_params['in_channels'] if a_params['in_channels'] > 0 else np.max(input_shape) * 2
            
            action = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=a_params['out_channels'],
                kernel_size=a_params['kernel_size'],
                stride=a_params['stride'],
                padding=a_params['padding']
            )

            # keep track of the convolution shapes
            layer_shapes.append(
                deconv2d_output_shape(
                    input_shape=layer_shapes[-1],
                    kernel_size=a_params['kernel_size'],
                    stride=a_params['stride'],
                    padding=a_params['padding'],
                    output_padding=0
                )
            )
        elif a['function'] == 'batchNorm2d':
            action = nn.BatchNorm2d(a_params['size'])

        elif a['function'] == 'relu':
            action = nn.LeakyReLU(negative_slope=a_params['negative_slope'])

        seq.append(action)
    return seq, layer_shapes

class Encoder(nn.Module):
    def __init__(self, latent_dim, input_shape, architecture):
        super(Encoder, self).__init__()        
        seq, layer_shapes = parse_architecture(input_shape, architecture)
        self.convolutions = nn.Sequential(*seq)

        self.flattened_size = layer_shapes[-1][0] * layer_shapes[-1][1] * architecture[-1]['params']['out_channels']
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        # self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

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

        seq, layer_shapes = parse_architecture(self.input_shape, architecture)
        self.seq = seq
        self.layer_shapes = layer_shapes
        self.deconvolutions = nn.Sequential(*seq)
        # self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # self.batch_norm1 = nn.BatchNorm2d(64)

        # self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=1, padding=1)
        # self.batch_norm2 = nn.BatchNorm2d(32)

        # self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), -1, self.input_shape[0], self.input_shape[1])

        z = self.deconvolutions(z)
        z = self.sigmoid(z)

        return z

class VAE(nn.Module):
    def __init__(self, params, input_shape):
        super(VAE, self).__init__()

        self.encoder = Encoder(latent_dim=params['latent_dim'], input_shape=input_shape, architecture=encoder_architecture)
        self.decoder = Decoder(latent_dim=params['latent_dim'], input_shape=input_shape, init_size=params['max_size'], architecture=decoder_architecture)
        self.sr_1 = SuperResolution()
        self.sr_2 = SuperResolution()
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

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
      