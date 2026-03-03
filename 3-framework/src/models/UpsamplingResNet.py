import torch.nn as nn
from .UpsamplingResidualBlock import UpsamplingResidualBlock

class UpsamplingResNet(nn.Module):
    def __init__(self, dropout_blocks=0.0, input_dim=256, rgb=True):
        """Decoder that upsamples from latent space to 3x32x32 image
        Mirror of encoder ResNet
        """
        super().__init__()

        # 256 -> 256x2x2
        self.reshape_channels = 256
        self.fc1 = nn.Linear(input_dim, self.reshape_channels*2*2)

        # 2x2 -> 4x4
        self.stageIV = nn.Sequential(
            UpsamplingResidualBlock(in_channels=256, out_channels=256, upsample=True, dropout=dropout_blocks),
            UpsamplingResidualBlock(in_channels=256, out_channels=256, upsample=False, dropout=dropout_blocks),
            UpsamplingResidualBlock(in_channels=256, out_channels=256, upsample=False, dropout=dropout_blocks),
            UpsamplingResidualBlock(in_channels=256, out_channels=256, upsample=False, dropout=dropout_blocks)
        )

        # 4x4 -> 8x8
        self.stageIII = nn.Sequential(
            UpsamplingResidualBlock(in_channels=256, out_channels=128, upsample=True, dropout=dropout_blocks),
            UpsamplingResidualBlock(in_channels=128, out_channels=128, upsample=False, dropout=dropout_blocks),
            UpsamplingResidualBlock(in_channels=128, out_channels=128, upsample=False, dropout=dropout_blocks),
            UpsamplingResidualBlock(in_channels=128, out_channels=128, upsample=False, dropout=dropout_blocks)
        )

        # 8x8 -> 16x16
        self.stageII = nn.Sequential(
            UpsamplingResidualBlock(in_channels=128, out_channels=64, upsample=True, dropout=dropout_blocks),
            UpsamplingResidualBlock(in_channels=64, out_channels=64, upsample=False, dropout=dropout_blocks),
            UpsamplingResidualBlock(in_channels=64, out_channels=64, upsample=False, dropout=dropout_blocks),
            UpsamplingResidualBlock(in_channels=64, out_channels=64, upsample=False, dropout=dropout_blocks)
        )

        # 16x16 -> 32x32
        self.stageI = nn.Sequential(
            UpsamplingResidualBlock(in_channels=64, out_channels=32, upsample=True, dropout=dropout_blocks),
            UpsamplingResidualBlock(in_channels=32, out_channels=32, upsample=False, dropout=dropout_blocks),
            UpsamplingResidualBlock(in_channels=32, out_channels=32, upsample=False, dropout=dropout_blocks),
            UpsamplingResidualBlock(in_channels=32, out_channels=32, upsample=False, dropout=dropout_blocks)
        )

        # Final conv to get 3 channels
        self.final_conv = nn.Conv2d(32, 3 if rgb else 1, kernel_size=3, padding=1)
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((28, 28)) #for MNIST, to get 28x28 output instead of 32x32
        self.final_activation = nn.Sigmoid()  # or Tanh() depending on your data normalization


    def forward(self, x):
        # x shape: (batch, input_dim)
        out = self.fc1(x)
        out = out.view(-1, self.reshape_channels, 2, 2)  # Reshape to 256x2x2
        
        out = self.stageIV(out)
        out = self.stageIII(out)
        out = self.stageII(out)
        out = self.stageI(out)
        
        out = self.final_conv(out)
        #out = self.adaptive_pool(out) # for MNIST, to get 28x28 output instead of 32x32
        out = self.final_activation(out)
        
        return out  # Output: (batch, 3, 32, 32)
