import torch.nn as nn
from .ResNet import ResNetMNIST, ResNetCIFAR10
from .UpsamplingResNet import UpsamplingResNetMNIST, UpsamplingResNetCIFAR10

class GANBase(nn.Module):
    def __init__(self, generator, discriminator, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = generator
        self.generator.final_activation = nn.Tanh()  # Override to Tanh for GANs, and make sure (in the dataloader) that images are normalized to [-1, 1]
        self.discriminator = discriminator

    def forward(self, z):
        return self.generator(z)


class GANMNIST(GANBase):
    def __init__(self, dropout=0.0, latent_dim=100):
        generator = UpsamplingResNetMNIST(dropout=dropout, input_dim=latent_dim)
        discriminator = ResNetMNIST(dropout=dropout, output_dim=1, norm="group")
        super().__init__(generator, discriminator, latent_dim=latent_dim)


class GANCIFAR10(GANBase):
    def __init__(self, dropout=0.0, latent_dim=100):      
        generator = UpsamplingResNetCIFAR10(dropout=dropout, input_dim=latent_dim)
        discriminator = ResNetCIFAR10(dropout=dropout, output_dim=1, norm="group")
        super().__init__(generator, discriminator, latent_dim=latent_dim)
