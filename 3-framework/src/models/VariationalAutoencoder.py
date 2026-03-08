import torch
import torch.nn as nn
from .ResNet import ResNetMNIST, ResNetCIFAR10
from .UpsamplingResNet import UpsamplingResNetMNIST, UpsamplingResNetCIFAR10

class VariationalAutoencoderBase(nn.Module):
    def __init__(self, encoder, decoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        mu = h[:, :self.latent_dim]
        log_var = h[:, self.latent_dim:]
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)

        if self.training:
            # Reparameterization trick
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # Use mean for inference

        reconstructed = self.decode(z)
        return mu, log_var, reconstructed


class VariationalAutoencoderMNIST(VariationalAutoencoderBase):
    def __init__(self, dropout=0.0, latent_dim=100):
        encoder = ResNetMNIST(dropout=dropout, output_dim=2*latent_dim) # Output both mean and log-variance
        decoder = UpsamplingResNetMNIST(dropout=dropout, input_dim=latent_dim)
        super().__init__(encoder, decoder, latent_dim=latent_dim)


class VariationalAutoencoderCIFAR10(VariationalAutoencoderBase):
    def __init__(self, dropout=0.0, latent_dim=100):
        encoder = ResNetCIFAR10(dropout=dropout, output_dim=2*latent_dim)
        decoder = UpsamplingResNetCIFAR10(dropout=dropout, input_dim=latent_dim)
        super().__init__(encoder, decoder, latent_dim=latent_dim)
