import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,)
        device = t.device
        half = self.dim // 2

        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]

        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

        self.residual = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.block1(x)

        time_emb = self.time_mlp(t_emb)
        h = h + time_emb[:, :, None, None]

        h = self.block2(h)
        return h + self.residual(x)


class UNetMNIST(nn.Module):
    def __init__(self, in_channels=1, base=64, time_dim=128):
        super().__init__()

        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # encoder
        self.enc1 = ResBlock(in_channels, base, time_dim)
        self.down1 = nn.Conv2d(base, base, 4, stride=2, padding=1)   # 28 -> 14

        self.enc2 = ResBlock(base, base * 2, time_dim)
        self.down2 = nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1)  # 14 -> 7

        # bottleneck
        self.mid = ResBlock(base * 2, base * 2, time_dim)

        # decoder
        self.up1 = nn.ConvTranspose2d(base * 2, base * 2, 4, stride=2, padding=1)  # 7 -> 14
        self.dec1 = ResBlock(base * 4, base, time_dim)

        self.up2 = nn.ConvTranspose2d(base, base, 4, stride=2, padding=1)  # 14 -> 28
        self.dec2 = ResBlock(base * 2, base, time_dim)

        self.final = nn.Conv2d(base, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x1 = self.enc1(x, t_emb)
        x2 = self.down1(x1)

        x2 = self.enc2(x2, t_emb)
        x3 = self.down2(x2)

        x3 = self.mid(x3, t_emb)

        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.dec1(x, t_emb)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec2(x, t_emb)

        return self.final(x)   # predicted noise ε


class UNetCIFAR10(nn.Module):
    def __init__(self, in_channels=3, base=128, time_dim=128):
        super().__init__()

        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # encoder
        self.enc1 = ResBlock(in_channels, base, time_dim)
        self.down1 = nn.Conv2d(base, base, 4, stride=2, padding=1)      # 32x32 -> 16x16

        self.enc2 = ResBlock(base, base * 2, time_dim)
        self.down2 = nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1) # 16x16 -> 8x8

        self.enc3 = ResBlock(base * 2, base * 4, time_dim)
        self.down3 = nn.Conv2d(base * 4, base * 4, 4, stride=2, padding=1) # 8x8 -> 4x4

        # bottleneck
        self.mid = ResBlock(base * 4, base * 4, time_dim)

        # decoder
        self.up1 = nn.ConvTranspose2d(base * 4, base * 4, 4, stride=2, padding=1) # 4x4 -> 8x8
        self.dec1 = ResBlock(base * 8, base * 2, time_dim)

        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 4, stride=2, padding=1) # 8x8 -> 16x16
        self.dec2 = ResBlock(base * 4, base, time_dim)

        self.up3 = nn.ConvTranspose2d(base, base, 4, stride=2, padding=1)        # 16x16 -> 32x32
        self.dec3 = ResBlock(base * 2, base, time_dim)

        self.final = nn.Conv2d(base, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Down
        x1 = self.enc1(x, t_emb)
        x2 = self.down1(x1)

        x2 = self.enc2(x2, t_emb)
        x3 = self.down2(x2)

        x3 = self.enc3(x3, t_emb)
        x4 = self.down3(x3)

        # Mid
        x4 = self.mid(x4, t_emb)

        # Up
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec1(x, t_emb)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x, t_emb)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec3(x, t_emb)

        return self.final(x)


class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()

        self.model = model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_hat', alpha_hat)
        self.register_buffer('sqrt_alpha_hat', torch.sqrt(alpha_hat))
        self.register_buffer(
            'sqrt_one_minus_alpha_hat',
            torch.sqrt(1.0 - alpha_hat)
        )

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alpha_hat[t][:, None, None, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_hat[t][:, None, None, None]

        xt = sqrt_ab * x0 + sqrt_1mab * noise
        return xt, noise

    def loss(self, x0):
        B = x0.size(0)
        device = x0.device

        # sample timestep uniformly
        t = torch.randint(0, self.timesteps, (B,), device=device)

        # forward diffusion
        xt, noise = self.q_sample(x0, t)

        # predict noise
        pred_noise = self.model(xt, t)

        # Ho et al. simple objective
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, n, device, channels=3, x=32, y=32):
        x = torch.randn(n, channels, x, y, device=device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)

            eps = self.model(x, t_batch)

            alpha = self.alphas[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.betas[t]

            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)

            x = (
                1 / torch.sqrt(alpha)
            ) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * eps
            ) + torch.sqrt(beta) * z

        return x.clamp(-1, 1)


import torchvision
from torchvision.utils import make_grid
import mlflow

# todo: unify with log_gan_images. Implmement "Log images" in mlflow or utils; then call it from both train_gan and train_ddpm - do generation in the training function, not here

def log_ddpm_images(model, epoch, device, num_images=100):

    model.eval()

    with torch.no_grad():
        generated_images = model.sample(num_images, channels=3, x=32, y=32, device=device)

    grid = make_grid(
        generated_images,
        nrow=10,
        padding=0,
        normalize=True,
        value_range=(-1, 1)
    )

    pil_image = torchvision.transforms.functional.to_pil_image(grid)

    mlflow.log_image(pil_image, f"images/generated_epoch_{epoch:03d}.png")

    model.train()
