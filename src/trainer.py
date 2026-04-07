"""
src/trainer.py — AttGAN training and checkpoint logic.

The Trainer class encapsulates the full training loop:
  • alternating D / G updates per batch
  • periodic sample visualisation and checkpoint saving
  • checkpoint resume support

Usage:
    from src.trainer import Trainer
    trainer = Trainer(enc, gen, dis, train_loader, test_loader, cfg, device)
    trainer.train()
"""

import torch
from tqdm import tqdm

from src.losses import AttGANLoss, build_optimizers
from src.utils  import save_checkpoint, load_checkpoint, visualise_samples


class Trainer:
    """
    Manages the AttGAN training loop.

    Args:
        enc, gen, dis : model instances (already on device)
        train_loader  : DataLoader for training split
        test_loader   : DataLoader for test split (used for visualisation)
        cfg           : Config instance
        device        : torch.device
    """

    def __init__(self, enc, gen, dis,
                 train_loader, test_loader,
                 cfg, device: torch.device):
        self.enc = enc
        self.gen = gen
        self.dis = dis
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.cfg    = cfg
        self.device = device

        self.criterion = AttGANLoss(device)
        self.optim_G, self.optim_D = build_optimizers(enc, gen, dis, cfg)

        # Fixed test batch — same images every epoch for consistent visuals
        self._fix_test_batch()

        self.g_losses: list[float] = []
        self.d_losses: list[float] = []

    # ── Setup ─────────────────────────────────────────────────────────

    def _fix_test_batch(self):
        imgs, attrs = next(iter(self.test_loader))
        self.test_imgs  = imgs[:8].to(self.device)
        self.test_attrs = attrs[:8].to(self.device)

    # ── One epoch ─────────────────────────────────────────────────────

    def _train_epoch(self) -> tuple[float, float]:
        self.enc.train(); self.gen.train(); self.dis.train()
        g_sum, d_sum, n = 0.0, 0.0, 0
        cfg = self.cfg
        crit = self.criterion

        for step, (imgs, attrs) in enumerate(
            tqdm(self.train_loader, desc="  batches", leave=False)
        ):
            imgs  = imgs.to(self.device)
            attrs = attrs.to(self.device)
            B     = imgs.size(0)

            # Sample target attributes by shuffling the batch
            perm         = torch.randperm(B)
            target_attrs = attrs[perm]

            # ══ 1. Train Discriminator ══
            self.optim_D.zero_grad()

            adv_real, cls_real = self.dis(imgs)
            loss_D = (
                crit.d_adv_real(adv_real)
                + crit.d_cls(cls_real, attrs) * cfg.LAMBDA_CLS_D
            )

            # Generate fakes without tracking G gradients
            with torch.no_grad():
                z     = self.enc(imgs)
                fakes = self.gen(z, target_attrs)

            adv_fake, _ = self.dis(fakes.detach())
            loss_D += crit.d_adv_fake(adv_fake)

            loss_D.backward()
            self.optim_D.step()

            # ══ 2. Train Generator (Enc + Dec) ══
            self.optim_G.zero_grad()

            z = self.enc(imgs)

            # Reconstruction: same attributes → should recover input
            rec  = self.gen(z, attrs)
            loss_rec = crit.g_rec(rec, imgs)

            # Translation: target attributes
            fakes = self.gen(z, target_attrs)
            adv_fake, cls_fake = self.dis(fakes)

            loss_G = (
                crit.g_adv(adv_fake)
                + crit.g_cls(cls_fake, target_attrs) * cfg.LAMBDA_CLS_G
                + loss_rec * cfg.LAMBDA_REC
            )
            loss_G.backward()
            self.optim_G.step()

            g_sum += loss_G.item()
            d_sum += loss_D.item()
            n     += 1

            if (step + 1) % cfg.LOG_EVERY_STEPS == 0:
                tqdm.write(
                    f"    step {step+1:>4}  "
                    f"G: {loss_G.item():.4f}  D: {loss_D.item():.4f}"
                )

        return g_sum / n, d_sum / n

    # ── Full training run ─────────────────────────────────────────────

    def train(self, resume_path: str | None = None):
        """
        Run the full training loop.

        Args:
            resume_path : optional path to a checkpoint .pt file to resume from
        """
        start_epoch = 0
        if resume_path is not None:
            start_epoch = load_checkpoint(
                resume_path, self.enc, self.gen, self.dis
            )

        cfg = self.cfg
        print(f"\n[trainer] Training for {cfg.N_EPOCHS} epochs "
              f"starting at epoch {start_epoch + 1}")

        for epoch in range(start_epoch + 1, cfg.N_EPOCHS + 1):
            g_loss, d_loss = self._train_epoch()
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss)

            print(f"Epoch [{epoch:>3}/{cfg.N_EPOCHS}]  "
                  f"G: {g_loss:.4f}  D: {d_loss:.4f}")

            if epoch % cfg.SAVE_EVERY == 0 or epoch == 1:
                visualise_samples(
                    self.enc, self.gen,
                    self.test_imgs, self.test_attrs,
                    epoch, cfg,
                )
                save_checkpoint(
                    self.enc, self.gen, self.dis, epoch, cfg
                )

        print("\n[trainer] Training complete ✓")

        # Compute and persist metrics
        if cfg.COMPUTE_METRICS:
            from src.metrics import compute_metrics
            scores = compute_metrics(
                self.enc, self.gen, self.test_loader, cfg, self.device
            )
            self._save_metrics(scores)

        return self.g_losses, self.d_losses


    def _save_metrics(self, scores: dict):
        """Write FID / DACID + loss history to a JSON file in RESULTS_DIR."""
        import json
        payload = {
            "experiment": self.cfg.EXPERIMENT_NAME,
            "fid":        scores.get("fid"),
            "dacid":      scores.get("dacid"),
            "g_losses":   self.g_losses,
            "d_losses":   self.d_losses,
        }
        path = self.cfg.RESULTS_DIR / "metrics.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[trainer] Metrics saved → {path}")
