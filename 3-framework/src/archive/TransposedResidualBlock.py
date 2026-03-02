import torch.nn as nn

class TransposedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        """ 
            Upsampling residual block for decoder.
            stride=2 means upsample by 2x (opposite of encoder)
        """
        super().__init__()

        # Skip connection: Fixes the tensor mismatch bug and artifacting
        if in_channels != out_channels or stride != 1:
            layers = []
            if stride != 1:
                # Upsample spatially first, then adjust channels (prevents sparse grid artifacts)
                layers.append(nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False))
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ])
            self.skip = nn.Sequential(*layers)
        else:
            self.skip = nn.Identity()

        # Main path with upsampling
        if stride != 1:
            # Using ConvTranspose2d for the spatial expansion
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, 
                                            stride=stride, padding=1, output_padding=stride-1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Swapped to standard Conv2d since stride=1 (more efficient)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        skip = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += skip  # In-place addition is slightly more memory efficient
        out = self.relu(out)
        
        if self.dropout.p > 0:
            out = self.dropout(out)
            
        return out