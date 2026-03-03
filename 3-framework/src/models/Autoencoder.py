import torch.nn as nn
from .ResNet import ResNet
from .UpsamplingResNet import UpsamplingResNet

class Autoencoder(nn.Module):
    def __init__(self, dropout=0.0, latent_dim=10, rgb=True):
        super().__init__()
        self.encoder = ResNet(dropout_blocks=dropout, nodes_final_layer=latent_dim, rgb=rgb)
        self.decoder = UpsamplingResNet(dropout_blocks=dropout, input_dim=latent_dim, rgb=rgb)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed
