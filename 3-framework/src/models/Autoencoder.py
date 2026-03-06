import torch.nn as nn
from .ResNet import ResNetMNIST, ResNetCIFAR10
from .UpsamplingResNet import UpsamplingResNetMNIST, UpsamplingResNetCIFAR10

class AutoencoderBase(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed


class AutoencoderMNIST(AutoencoderBase):
    def __init__(self, dropout=0.0, latent_dim=100):
        self.latent_dim = latent_dim
        self.encoder = ResNetMNIST(dropout=dropout, output_dim=latent_dim)
        self.decoder = UpsamplingResNetMNIST(dropout=dropout, input_dim=latent_dim)
        super().__init__(self.encoder, self.decoder)


class AutoencoderCIFAR10(AutoencoderBase):
    def __init__(self, dropout=0.0, latent_dim=100):
        self.latent_dim = latent_dim
        self.encoder = ResNetCIFAR10(dropout=dropout, output_dim=latent_dim)
        self.decoder = UpsamplingResNetCIFAR10(dropout=dropout, input_dim=latent_dim)
        super().__init__(self.encoder, self.decoder)
