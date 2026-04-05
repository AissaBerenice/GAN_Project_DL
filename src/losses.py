"""
src/losses.py — Loss functions and optimizers for AttGAN.

AttGAN uses three losses working in concert:

  L_adv  (MSELoss / LSGAN)
    Makes generated images indistinguishable from real ones.
    LSGAN is chosen over vanilla GAN BCE because it produces
    more stable gradients and avoids vanishing-gradient issues.

  L_cls  (BCEWithLogitsLoss)
    Ensures the discriminator's attribute head correctly classifies
    both real images (supervised signal for D) and generated images
    (attribute accuracy signal for G).

  L_rec  (L1Loss)
    When the generator receives the *original* attributes as input,
    the output should reconstruct the input image exactly. This loss
    preserves identity and non-edited regions.

Total losses:
  L_D = L_adv_real + L_adv_fake + λ_cls_D · L_cls_real
  L_G = L_adv_G    + λ_cls_G · L_cls_G + λ_rec · L_rec

Usage:
    from src.losses import AttGANLoss, build_optimizers
"""

import torch
import torch.nn as nn
import torch.optim as optim


class AttGANLoss:
    """
    Wraps the three loss functions used in AttGAN.
    Labels for LSGAN: real = 1.0, fake = 0.0.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.adv = nn.MSELoss()             # LSGAN adversarial
        self.cls = nn.BCEWithLogitsLoss()   # attribute classification
        self.rec = nn.L1Loss()              # pixel reconstruction

    # ── label helpers ────────────────────────────────────────────────
    def ones(self, size: int) -> torch.Tensor:
        return torch.ones(size, 1, device=self.device)

    def zeros(self, size: int) -> torch.Tensor:
        return torch.zeros(size, 1, device=self.device)

    # ── {-1,+1} → {0,1} for BCE ──────────────────────────────────────
    @staticmethod
    def to_binary(a: torch.Tensor) -> torch.Tensor:
        """Convert bipolar {-1,+1} attribute tensor to {0,1} for BCE."""
        return (a + 1) / 2

    # ── Discriminator losses ─────────────────────────────────────────
    def d_adv_real(self, adv_real: torch.Tensor) -> torch.Tensor:
        return self.adv(adv_real, self.ones(adv_real.size(0)))

    def d_adv_fake(self, adv_fake: torch.Tensor) -> torch.Tensor:
        return self.adv(adv_fake, self.zeros(adv_fake.size(0)))

    def d_cls(self, cls_logits: torch.Tensor,
              attrs: torch.Tensor) -> torch.Tensor:
        return self.cls(cls_logits, self.to_binary(attrs))

    # ── Generator losses ─────────────────────────────────────────────
    def g_adv(self, adv_fake: torch.Tensor) -> torch.Tensor:
        """Generator wants D to output 1 for fakes (fool D)."""
        return self.adv(adv_fake, self.ones(adv_fake.size(0)))

    def g_cls(self, cls_logits: torch.Tensor,
              target_attrs: torch.Tensor) -> torch.Tensor:
        return self.cls(cls_logits, self.to_binary(target_attrs))

    def g_rec(self, rec_img: torch.Tensor,
              real_img: torch.Tensor) -> torch.Tensor:
        return self.rec(rec_img, real_img)


def build_optimizers(enc, gen, dis, cfg):
    """
    Create one optimizer for the generator (Enc + Dec parameters)
    and one for the discriminator, both using Adam.

    Returns:
        (optim_G, optim_D)
    """
    optim_G = optim.Adam(
        list(enc.parameters()) + list(gen.parameters()),
        lr=cfg.LR,
        betas=(cfg.BETA1, cfg.BETA2),
    )
    optim_D = optim.Adam(
        dis.parameters(),
        lr=cfg.LR,
        betas=(cfg.BETA1, cfg.BETA2),
    )
    return optim_G, optim_D