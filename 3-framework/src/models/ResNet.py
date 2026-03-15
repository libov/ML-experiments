import torch.nn as nn
from .ResidualBlock import ResidualBlock

class ResNetBase(nn.Module):
    """
        Base class for ResNet architecture.
        Should be subclassed for specific datasets (e.g., CIFAR-10, MNIST, ImageNet) to set appropriate input channels and output dimensions.
        Parameters:
            in_channels: number of channels in the input image (3 for RGB, 1 for grayscale)
            ch: channel count at the output of each stage (stem, stageI, stageII, stageIII, stageIV)
            dropout: dropout rate to apply in each residual block
            output_dim: number of output classes or embedding dimension
            stem: optional stem module to replace the default convolutional layer (useful for ImageNet)
    """
    def __init__(self, in_channels=3, ch=[16, 32, 64, 128, 256], dropout=0.0, output_dim=10, stem=None, norm = "batch"):
        super().__init__()

        self.stem = stem if stem is not None else nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = ch[0], kernel_size = 3, padding = 'same', bias=False),
            nn.BatchNorm2d(ch[0]) if norm == "batch" else nn.GroupNorm(1, ch[0]) if norm == "group" else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.stageI = nn.Sequential(
            ResidualBlock(in_channels = ch[0], out_channels = ch[1], stride=1, dropout = dropout, norm = norm),
            ResidualBlock(in_channels = ch[1], out_channels = ch[1], stride=1, dropout = dropout, norm = norm),
            ResidualBlock(in_channels = ch[1], out_channels = ch[1], stride=1, dropout = dropout, norm = norm),
            ResidualBlock(in_channels = ch[1], out_channels = ch[1], stride=1, dropout = dropout, norm = norm)
        )

        self.stageII = nn.Sequential(
            ResidualBlock(in_channels = ch[1], out_channels = ch[2], stride=2, dropout = dropout, norm = norm),
            ResidualBlock(in_channels = ch[2], out_channels = ch[2], stride=1, dropout = dropout, norm = norm),
            ResidualBlock(in_channels = ch[2], out_channels = ch[2], stride=1, dropout = dropout, norm = norm),
            ResidualBlock(in_channels = ch[2], out_channels = ch[2], stride=1, dropout = dropout, norm = norm)
        )

        self.stageIII = nn.Sequential(
            ResidualBlock(in_channels = ch[2], out_channels = ch[3], stride=2, dropout = dropout, norm = norm),
            ResidualBlock(in_channels = ch[3], out_channels = ch[3], stride=1, dropout = dropout, norm = norm),
            ResidualBlock(in_channels = ch[3], out_channels = ch[3], stride=1, dropout = dropout, norm = norm),
            ResidualBlock(in_channels = ch[3], out_channels = ch[3], stride=1, dropout = dropout, norm = norm)
        )

        self.stageIV = nn.Sequential(
            ResidualBlock(in_channels = ch[3], out_channels = ch[4], stride=2, dropout = dropout, norm = norm),
            ResidualBlock(in_channels = ch[4], out_channels = ch[4], stride=1, dropout = dropout, norm = norm),
            ResidualBlock(in_channels = ch[4], out_channels = ch[4], stride=1, dropout = dropout, norm = norm),
            ResidualBlock(in_channels = ch[4], out_channels = ch[4], stride=1, dropout = dropout, norm = norm)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch[4], output_dim)
        )

    def forward(self, x):

        out = self.stem(x)

        out = self.stageI(out)
        out = self.stageII(out)
        out = self.stageIII(out)
        out = self.stageIV(out)

        out = self.head(out)

        return out


class ResNetMNIST(ResNetBase):
    def __init__(self, dropout=0.0, output_dim=10, norm = "batch"):
        super().__init__(in_channels=1, ch=[16, 32, 64, 128, 256], dropout=dropout, output_dim=output_dim, norm = norm)


class ResNetCIFAR10(ResNetBase):
    def __init__(self, dropout=0.0, output_dim=10, norm = "batch"):
        super().__init__(in_channels=3, ch=[16, 32, 64, 128, 256], dropout=dropout, output_dim=output_dim, norm = norm)


class ResNetImageNet(ResNetBase):
    def __init__(self, dropout=0.0, output_dim=1000, norm = "batch"):
        in_channels = 3
        ch = [64, 64, 128, 256, 512]
        stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ch[0]) if norm == "batch" else nn.GroupNorm(1, ch[0]) if norm == "group" else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # note that in_channels is not passed because it is relevant only for the stem - which we override here and pass explicitly as the last argument
        super().__init__(in_channels=None, ch=ch, dropout=dropout, output_dim=output_dim, stem=stem, norm = norm)
