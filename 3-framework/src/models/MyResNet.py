import torch.nn as nn
from .ResidualBlock import ResidualBlock

class MyResNet(nn.Module):
    def __init__(self, dropout_blocks=0.0):
        super().__init__()

        # 3x32x32 -> 16x32x32
        self.initial_conv =  nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 'same') 
        self.initial_bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 32x32 -> 32x32
        self.stageI = nn.Sequential(
            ResidualBlock(in_channels = 16, out_channels = 32, stride=1, dropout = dropout_blocks),
            ResidualBlock(in_channels = 32, out_channels = 32, stride=1, dropout = dropout_blocks),
            ResidualBlock(in_channels = 32, out_channels = 32, stride=1, dropout = dropout_blocks),
            ResidualBlock(in_channels = 32, out_channels = 32, stride=1, dropout = dropout_blocks)
        )

        # 32x32 -> 16x16
        self.stageII = nn.Sequential(
            ResidualBlock(in_channels = 32, out_channels = 64, stride=2, dropout = dropout_blocks),
            ResidualBlock(in_channels = 64, out_channels = 64, stride=1, dropout = dropout_blocks),
            ResidualBlock(in_channels = 64, out_channels = 64, stride=1, dropout = dropout_blocks),
            ResidualBlock(in_channels = 64, out_channels = 64, stride=1, dropout = dropout_blocks)
        )

        # 16x16 -> 8x8
        self.stageIII = nn.Sequential(
            ResidualBlock(in_channels = 64, out_channels = 128, stride=2, dropout = dropout_blocks),
            ResidualBlock(in_channels = 128, out_channels = 128, stride=1, dropout = dropout_blocks),
            ResidualBlock(in_channels = 128, out_channels = 128, stride=1, dropout = dropout_blocks),
            ResidualBlock(in_channels = 128, out_channels = 128, stride=1, dropout = dropout_blocks)
        )

        # 8x8 -> 4x4
        self.stageIV = nn.Sequential(
            ResidualBlock(in_channels = 128, out_channels = 256, stride=2, dropout = dropout_blocks),
            ResidualBlock(in_channels = 256, out_channels = 256, stride=1, dropout = dropout_blocks),
            ResidualBlock(in_channels = 256, out_channels = 256, stride=1, dropout = dropout_blocks),
            ResidualBlock(in_channels = 256, out_channels = 256, stride=1, dropout = dropout_blocks)
        )

        self.adaptivepool = nn.AdaptiveAvgPool2d(1) # 256x4x4  -> 256x1x1

        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = self.relu(out)

        out = self.stageI(out)
        out = self.stageII(out)
        out = self.stageIII(out)
        out = self.stageIV(out)

        out = self.adaptivepool(out)
        
        out = self.flatten(out)
        out = self.fc2(out)

        return out

