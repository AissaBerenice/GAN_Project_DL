"""
src/models.py — AttGAN neural network components.

Three modules implement the full AttGAN architecture:

  Encoder (G_enc)
    image (3×128×128) → latent z (ENC_DIM*8 × 4 × 4)
    Five strided convolutions halve spatial dims: 128→64→32→16→8→4.

  Generator / Decoder (G_dec)
    (z, target_attrs) → edited image (3×128×128)
    Target attribute vector is tiled spatially and concatenated to z,
    then decoded with five transposed convolutions.

  Discriminator (D)
    image → (adv_logit, cls_logits)
    Shared feature extractor feeds two heads:
      adv_head → scalar   (LSGAN real / fake)
      cls_head → n_attrs  (attribute classification)

Usage:
    from src.models import Encoder, Generator, Discriminator
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → InstanceNorm → LeakyReLU (used in Encoder / Discriminator)."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 4, stride: int = 2, pad: int = 1,
                 norm: bool = True, act: str = "leaky"):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=not norm)
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        if act == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif act == "relu":
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """ConvTranspose2d → InstanceNorm → ReLU (used in Generator)."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 4, stride: int = 2, pad: int = 1,
                 norm: bool = True, act: str = "relu"):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, pad, bias=not norm)
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        if act == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif act == "tanh":
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────
# Encoder  G_enc
# ─────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Compresses a face image into a bottleneck latent tensor.

    Input  : (B, 3, 128, 128)
    Output : (B, dim*8, 4, 4)   →  512 × 4 × 4 when dim=64
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3,       dim,    norm=False),   # 128 → 64
            ConvBlock(dim,     dim*2),                 # 64  → 32
            ConvBlock(dim*2,   dim*4),                 # 32  → 16
            ConvBlock(dim*4,   dim*8),                 # 16  → 8
            ConvBlock(dim*8,   dim*8),                 # 8   → 4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────
# Generator / Decoder  G_dec
# ─────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Decodes a latent tensor + target attribute vector into an edited image.

    Attribute injection:
        a  (B, n_attrs) is expanded to (B, n_attrs, 4, 4) and concatenated
        with z along the channel dimension before decoding. This allows
        every spatial position to receive the full attribute condition.

    Input  : z  (B, dim*8, 4, 4),  a  (B, n_attrs)
    Output : (B, 3, 128, 128)  in range [−1, 1]
    """

    def __init__(self, n_attrs: int, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            UpBlock(dim*8 + n_attrs,  dim*8),   # 4   → 8
            UpBlock(dim*8,            dim*4),    # 8   → 16
            UpBlock(dim*4,            dim*2),    # 16  → 32
            UpBlock(dim*2,            dim),      # 32  → 64
            UpBlock(dim,              3,   norm=False, act="tanh"),  # 64 → 128
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h, w   = z.shape[2], z.shape[3]
        a_tile = a.view(a.size(0), -1, 1, 1).expand(-1, -1, h, w)
        return self.net(torch.cat([z, a_tile], dim=1))


# ─────────────────────────────────────────────────────────────────────
# Discriminator  D
# ─────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Multi-task discriminator with shared feature extractor.

    Two output heads:
      adv_head → (B, 1)        LSGAN real/fake scalar
      cls_head → (B, n_attrs)  attribute classification logits

    The attribute classification head is the key differentiator of
    AttGAN over a plain GAN — it provides gradient signal that steers
    the generator to produce the *correct* target attributes.

    Input  : (B, 3, 128, 128)
    Output : adv (B, 1),  cls (B, n_attrs)
    """

    def __init__(self, n_attrs: int, dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,      dim,    norm=False),  # 128 → 64
            ConvBlock(dim,    dim*2),                # 64  → 32
            ConvBlock(dim*2,  dim*4),                # 32  → 16
            ConvBlock(dim*4,  dim*8),                # 16  → 8
            ConvBlock(dim*8,  dim*8),                # 8   → 4
        )
        # Head 1 — adversarial (4×4 → 1×1 → scalar)
        self.adv_head = nn.Conv2d(dim*8, 1,       kernel_size=4, stride=1, padding=0)
        # Head 2 — attribute classifier (4×4 → 1×1 → n_attrs)
        self.cls_head = nn.Conv2d(dim*8, n_attrs, kernel_size=4, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        adv  = self.adv_head(feat).view(x.size(0), -1)
        cls  = self.cls_head(feat).view(x.size(0), -1)
        return adv, cls


# ─────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_models(cfg, device: torch.device):
    """Instantiate and return (Encoder, Generator, Discriminator) on device."""
    enc = Encoder(dim=cfg.ENC_DIM).to(device)
    gen = Generator(n_attrs=cfg.N_ATTRS, dim=cfg.DEC_DIM).to(device)
    dis = Discriminator(n_attrs=cfg.N_ATTRS, dim=cfg.DIS_DIM).to(device)

    print("[models] Parameters:")
    print(f"  Encoder      : {count_parameters(enc) / 1e6:.2f} M")
    print(f"  Generator    : {count_parameters(gen) / 1e6:.2f} M")
    print(f"  Discriminator: {count_parameters(dis) / 1e6:.2f} M")

    return enc, gen, dis