import torch.nn as nn
from .UpsamplingResidualBlock import UpsamplingResidualBlock

class UpsamplingResNetBase(nn.Module):
    """
        Decoder that upsamples a latent space vector to an image.
        Approximate mirror of encoder ResNet.
        Base class, to be subclassed for specific datasets (e.g., CIFAR-10, MNIST) to set appropriate input dimensions and output channels.
        Parameters:
            input_dim: dimension of the input latent space vector
            initial_res: initial spatial resolution after the first fully connected layer
                         Typical values: 4 for 4x4 (CIFAR), 7 for 7x7 (ImageNet, MNIST)
            upsample: a list of booleans indicating whether to upsample in the respective stage (stageIV, stageIII, stageII, stageI)
            ch: channel count at the output of each stage (to_grid, stageIV, stageIII, stageII, stageI)
            out_channels: number of channels in the output image (3 for RGB, 1 for grayscale)
            dropout: dropout rate to apply in each residual block
            reverse_stem: optional "reverse stem" module to replace the default convolution without upsampling (useful for ImageNet)
    """
    def __init__(self, input_dim=100,
                 initial_res=4,
                 ch=[256, 256, 128, 64, 32],
                 upsample=[False, True, True, True],
                 out_channels=3,
                 dropout=0.0,
                 reverse_stem=None):
        super().__init__()

        self.to_grid = nn.Sequential(
            nn.Linear(input_dim, ch[0]*initial_res*initial_res),
            nn.Unflatten(dim=1, unflattened_size=(ch[0], initial_res, initial_res))
        )

        self.stageIV = nn.Sequential(
            UpsamplingResidualBlock(in_channels=ch[0], out_channels=ch[1], upsample=upsample[0], dropout=dropout),
            UpsamplingResidualBlock(in_channels=ch[1], out_channels=ch[1], upsample=False,       dropout=dropout),
            UpsamplingResidualBlock(in_channels=ch[1], out_channels=ch[1], upsample=False,       dropout=dropout),
            UpsamplingResidualBlock(in_channels=ch[1], out_channels=ch[1], upsample=False,       dropout=dropout)
        )

        self.stageIII = nn.Sequential(
            UpsamplingResidualBlock(in_channels=ch[1], out_channels=ch[2], upsample=upsample[1], dropout=dropout),
            UpsamplingResidualBlock(in_channels=ch[2], out_channels=ch[2], upsample=False,       dropout=dropout),
            UpsamplingResidualBlock(in_channels=ch[2], out_channels=ch[2], upsample=False,       dropout=dropout),
            UpsamplingResidualBlock(in_channels=ch[2], out_channels=ch[2], upsample=False,       dropout=dropout)
        )

        self.stageII = nn.Sequential(
            UpsamplingResidualBlock(in_channels=ch[2], out_channels=ch[3], upsample=upsample[2], dropout=dropout),
            UpsamplingResidualBlock(in_channels=ch[3], out_channels=ch[3], upsample=False,       dropout=dropout),
            UpsamplingResidualBlock(in_channels=ch[3], out_channels=ch[3], upsample=False,       dropout=dropout),
            UpsamplingResidualBlock(in_channels=ch[3], out_channels=ch[3], upsample=False,       dropout=dropout)
        )

        self.stageI = nn.Sequential(
            UpsamplingResidualBlock(in_channels=ch[3], out_channels=ch[4], upsample=upsample[3], dropout=dropout),
            UpsamplingResidualBlock(in_channels=ch[4], out_channels=ch[4], upsample=False,       dropout=dropout),
            UpsamplingResidualBlock(in_channels=ch[4], out_channels=ch[4], upsample=False,       dropout=dropout),
            UpsamplingResidualBlock(in_channels=ch[4], out_channels=ch[4], upsample=False,       dropout=dropout)
        )

        self.reverse_stem = reverse_stem if reverse_stem is not None else nn.Sequential(
            nn.Conv2d(in_channels=ch[4], out_channels=out_channels, kernel_size=3, padding=1),
        )

        # For Tanh, create an instance and override manually (see GAN.py for example)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):

        out = self.to_grid(x)
        
        out = self.stageIV(out)
        out = self.stageIII(out)
        out = self.stageII(out)
        out = self.stageI(out)
        
        out = self.reverse_stem(out)
        out = self.final_activation(out)
        
        return out


class UpsamplingResNetMNIST(UpsamplingResNetBase):
    def __init__(self, dropout=0.0, input_dim=100):
        super().__init__(input_dim=input_dim,
                         initial_res=7,
                         ch=[256, 256, 128, 64, 32],
                         upsample=[False, True, True, False],
                         out_channels=1,
                         dropout=dropout)


class UpsamplingResNetCIFAR10(UpsamplingResNetBase):
    def __init__(self, dropout=0.0, input_dim=100):
        super().__init__(input_dim=input_dim,
                         initial_res=4,
                         ch=[256, 256, 128, 64, 32],
                         upsample=[False, True, True, True],
                         out_channels=3,
                         dropout=dropout)


class UpsamplingResNetImageNet(UpsamplingResNetBase):
    def __init__(self, dropout=0.0, input_dim=100):
        ch = [512, 512, 256, 128, 64]
        out_channels = 3

        # 56x56 -> 112x112 -> 224x224
        reverse_stem = nn.Sequential(
            # First upsample: 56x56 to 112x112
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=ch[4], out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Second upsample: 112x112 to 224x224
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Final projection to RGB
            nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=3, padding=1),
        )

        super().__init__(input_dim=input_dim,
                         initial_res=7,
                         ch=ch,
                         upsample=[False, True, True, True],
                         out_channels=None,
                         dropout=dropout,
                         reverse_stem=reverse_stem)
