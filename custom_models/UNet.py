import torch
import torch.nn as nn
from global_settings import device
from utils.vision_utils import render_image_grid

class DoubleConv(nn.Module):
    """(Convolution -> ReLU) * 2 with optional BatchNorm."""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ])

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super(UNet, self).__init__()
        self.num_base_filters = base_filters
        self.encoder1 = DoubleConv(in_channels, base_filters)
        self.encoder2 = DoubleConv(base_filters, base_filters * 2)
        self.encoder3 = DoubleConv(base_filters * 2, base_filters * 4)

        self.middle = DoubleConv(base_filters * 4, base_filters * 4)

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2),
            DoubleConv(base_filters * 2, base_filters * 2)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2),
            DoubleConv(base_filters, base_filters)
        )
        
        # self.decoder1 = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.decoder1 = nn.ConvTranspose2d(base_filters, out_channels, kernel_size=2, stride=2)
        # self.decoder1 = nn.Linear(base_filters, out_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Optional Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        
        # TODO: change architecture to allow skip connections
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))

        # Middle block
        middle = self.middle(self.maxpool(enc3))

        dec3 = self.decoder3(self.dropout(middle))
        dec2 = self.decoder2(self.dropout(dec3))
        dec1 = self.decoder1(self.dropout(dec2))

        return dec1

def positional_encoding(t, model_dim):
    half_dim = model_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(t.device)  # (half_dim,)
    
    pos_enc = t[:, None] * emb[None, :]  # Shape: (batch_size, half_dim)
    pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)  # Shape: (batch_size, model_dim)
    
    return pos_enc


def forward_diffusion(x, t, model_dim, noise_level=1.3):
    # Apply positional encoding to time steps
    time_encoding = positional_encoding(t, 256).squeeze(-2)
    
    # Generate noise and scale by noise factor
    noise_factor = torch.sin(t) ** 2 * noise_level
    noise = torch.randn_like(x) * noise_factor
    
    return x + noise + time_encoding

