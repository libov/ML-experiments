import torch.nn as nn
from .ResNet import ResNetMNIST, ResNetCIFAR10
from .UpsamplingResNet import UpsamplingResNetMNIST, UpsamplingResNetCIFAR10

def weights_init(m):
    classname = m.__class__.__name__

    # Initialize Conv and Linear layers
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    # Initialize Normalization layers (BatchNorm, GroupNorm, InstanceNorm)
    elif classname.find('Norm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class GANBase(nn.Module):
    def __init__(self, generator, discriminator, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = generator
        self.generator.final_activation = nn.Tanh()  # Override to Tanh for GANs, and make sure (in the dataloader) that images are normalized to [-1, 1]
        self.discriminator = discriminator

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

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
