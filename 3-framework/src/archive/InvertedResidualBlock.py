import torch.nn as nn

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        """ Upsampling residual block for decoder.
            stride=2 means upsample by 2x (opposite of encoder)
        """
        super().__init__()

        # Skip connection: adjust channels and spatial dims if needed
        self.skip = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()

        # Main path with upsampling
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, 
                                        stride=stride, padding=1, output_padding=stride-1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, 
                                        stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        skip = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + skip
        out = self.relu(out)
        out = self.dropout(out)
        return out