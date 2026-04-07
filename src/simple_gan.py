"""
src/simple_gan.py — Simple unconditional DCGAN on CelebA.

Implements the baseline GAN before AttGAN is introduced.
Architecture follows Radford et al. (2015) DCGAN guidelines:
  • strided convolutions instead of pooling
  • batch normalisation in G and D (except first/last layers)
  • LeakyReLU in D, ReLU in G, Tanh at G output

This gives the project a clean progression:
    Simple GAN (unconditional) → AttGAN (conditional, attribute-guided)

Classes:
    SimpleGenerator     : noise → image
    SimpleDiscriminator : image → real/fake scalar

Functions:
    build_simple_models : instantiate both models and print param counts
    train_simple_gan    : full training loop returning loss history
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import denorm


# ─────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────

class SimpleGenerator(nn.Module):
    """
    DCGAN Generator: latent vector z → 3×64×64 image.

    Upsampling path (each ConvTranspose2d doubles spatial dims):
        4×4 → 8×8 → 16×16 → 32×32 → 64×64

    Args:
        latent_dim : dimension of input noise vector (default 100)
        dim        : base channel multiplier (default 64)
    """

    def __init__(self, latent_dim: int = 100, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # (latent_dim, 1, 1) → (dim*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, dim*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(dim*8),
            nn.ReLU(True),
            # → (dim*4, 8, 8)
            nn.ConvTranspose2d(dim*8, dim*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(True),
            # → (dim*2, 16, 16)
            nn.ConvTranspose2d(dim*4, dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            # → (dim, 32, 32)
            nn.ConvTranspose2d(dim*2, dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            # → (3, 64, 64)
            nn.ConvTranspose2d(dim, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SimpleDiscriminator(nn.Module):
    """
    DCGAN Discriminator: 3×64×64 image → real/fake scalar.

    Downsampling path:
        64×64 → 32×32 → 16×16 → 8×8 → 4×4 → scalar

    Args:
        dim : base channel multiplier (default 64)
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # No BN in first layer (per DCGAN paper)
            nn.Conv2d(3,      dim,   4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim,    dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim*2,  dim*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim*4,  dim*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4×4 → scalar
            nn.Conv2d(dim*8,  1,     4, 1, 0, bias=False),
            nn.Sigmoid(),   # vanilla GAN uses BCE + Sigmoid
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)


# ─────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────

def build_simple_models(latent_dim: int = 100, dim: int = 64,
                         device: torch.device = torch.device("cpu")):
    gen = SimpleGenerator(latent_dim=latent_dim, dim=dim).to(device)
    dis = SimpleDiscriminator(dim=dim).to(device)

    def _count(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("[simple_gan] Parameters:")
    print(f"  Generator      : {_count(gen)/1e6:.2f} M")
    print(f"  Discriminator  : {_count(dis)/1e6:.2f} M")

    return gen, dis


# ─────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────

def train_simple_gan(gen, dis, train_loader, cfg, device: torch.device,
                      latent_dim: int = 100):
    """
    Train a simple GAN using vanilla binary cross-entropy adversarial loss.

    Generator loss  : BCE(D(G(z)), 1)   — fool D
    Discriminator loss: BCE(D(x), 1) + BCE(D(G(z)), 0)

    Args:
        gen, dis      : SimpleGenerator and SimpleDiscriminator
        train_loader  : DataLoader for CelebA
        cfg           : Config instance
        device        : torch.device
        latent_dim    : dimension of noise input to G

    Returns:
        (g_losses, d_losses) — per-epoch average losses
    """
    criterion = nn.BCELoss()
    optim_G   = optim.Adam(gen.parameters(),
                            lr=cfg.LR, betas=(cfg.BETA1, cfg.BETA2))
    optim_D   = optim.Adam(dis.parameters(),
                            lr=cfg.LR, betas=(cfg.BETA1, cfg.BETA2))

    # Fixed noise for consistent visualisation across epochs
    fixed_z = torch.randn(64, latent_dim, 1, 1, device=device)

    g_losses, d_losses = [], []

    print(f"\n[simple_gan] Training {cfg.N_EPOCHS} epochs")

    for epoch in range(1, cfg.N_EPOCHS + 1):
        gen.train(); dis.train()
        g_sum = d_sum = n = 0.0

        for imgs, _ in tqdm(train_loader, desc=f"  Epoch {epoch}", leave=False):
            # CelebA images are 128×128 — downsample to 64×64 for DCGAN
            imgs = torch.nn.functional.interpolate(imgs, size=64)
            imgs = imgs.to(device)
            B    = imgs.size(0)

            real_label = torch.ones(B, 1, device=device)
            fake_label = torch.zeros(B, 1, device=device)

            # ── Train D ──
            optim_D.zero_grad()
            d_real = criterion(dis(imgs),                               real_label)
            z      = torch.randn(B, latent_dim, 1, 1, device=device)
            d_fake = criterion(dis(gen(z).detach()),                    fake_label)
            loss_D = d_real + d_fake
            loss_D.backward()
            optim_D.step()

            # ── Train G ──
            optim_G.zero_grad()
            z      = torch.randn(B, latent_dim, 1, 1, device=device)
            loss_G = criterion(dis(gen(z)),                             real_label)
            loss_G.backward()
            optim_G.step()

            g_sum += loss_G.item(); d_sum += loss_D.item(); n += 1

        g_avg, d_avg = g_sum / n, d_sum / n
        g_losses.append(g_avg); d_losses.append(d_avg)
        print(f"Epoch [{epoch:>3}/{cfg.N_EPOCHS}]  G: {g_avg:.4f}  D: {d_avg:.4f}")

        if epoch % cfg.SAVE_EVERY == 0 or epoch == 1:
            _save_simple_samples(gen, fixed_z, epoch, cfg)

    print("[simple_gan] Training complete ✓")
    return g_losses, d_losses


def _save_simple_samples(gen, fixed_z: torch.Tensor, epoch: int, cfg):
    """Save a grid of 64 generated images from the fixed noise vector."""
    import torchvision
    gen.eval()
    with torch.no_grad():
        imgs = gen(fixed_z).cpu()
    grid = torchvision.utils.make_grid(denorm(imgs), nrow=8, padding=2)
    path = cfg.RESULTS_DIR / f"simple_gan_epoch{epoch:03d}.png"
    torchvision.utils.save_image(grid, path)

    # Display inline (Colab / Jupyter)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    plt.title(f"Simple GAN — Epoch {epoch}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.show()
    print(f"[simple_gan] Saved → {path}")
    gen.train()