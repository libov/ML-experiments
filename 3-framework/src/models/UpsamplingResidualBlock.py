import torch.nn as nn

class UpsamplingResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True, dropout=0.0):
        """ 
            The 'Opposite' Residual Block for upsampling paths.
            Uses Bilinear Upsampling + Conv2d to avoid checkerboard artifacts.
        """
        super().__init__()
        
        # Spatial upsampling layer (applied to the whole block input)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else nn.Identity()
        
        # Skip connection: purely for channel adjustment after upsampling
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

        # Main convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 1. Scale up spatial dimensions first
        x = self.upsample(x)
        
        # 2. Process skip connection (adjusts channels if needed)
        skip = self.skip(x)
        
        # 3. Process main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 4. Residual addition and final activation
        out += skip  
        out = self.relu(out)
        
        if self.dropout.p > 0:
            out = self.dropout(out)
            
        return out