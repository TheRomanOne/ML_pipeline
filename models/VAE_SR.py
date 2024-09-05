import torch
import torch.nn as nn
from models.SuperRsolution import SuperResolution

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        self.flattened_size = 128 * 8

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 16)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)

        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 128, 8, 16)

        z = self.relu(self.batch_norm1(self.deconv1(z)))
        z = self.relu(self.batch_norm2(self.deconv2(z)))
        z = self.sigmoid(self.deconv3(z))

        return z

class VAE_SR(nn.Module):
    def __init__(self, latent_dim):
        super(VAE_SR, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
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
        # recon_x = ((recon_x / 2 + .5))
        return recon_x, mu, logvar

    def from_latent(self, x, enhance=1):
      x = self.relu(self.decoder(x))
      for _ in range(enhance):
        x = self.sr_2(self.sr_1(x))
      return x