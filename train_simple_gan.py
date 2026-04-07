"""
train_simple_gan.py — Entry point for Simple GAN training on CelebA.

Run from the repo root:
    python train_simple_gan.py
    python train_simple_gan.py --epochs 20
    python train_simple_gan.py --no-metrics

This trains an unconditional DCGAN on CelebA (resized to 64×64) and
serves as the baseline GAN before AttGAN is introduced in the project.
"""

import argparse
import json
import torch

from config          import Config
from src.dataset     import get_loaders
from src.simple_gan  import build_simple_models, train_simple_gan
from src.utils       import plot_losses


LATENT_DIM = 100   # noise vector dimension


def parse_args():
    p = argparse.ArgumentParser(description="Train Simple DCGAN on CelebA")
    p.add_argument("--epochs",      type=int,  default=None,
                   help="Override Config.N_EPOCHS")
    p.add_argument("--batch",       type=int,  default=None,
                   help="Override Config.BATCH_SIZE")
    p.add_argument("--no-metrics",  action="store_true",
                   help="Skip FID / DACID computation after training")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = Config()
    cfg.EXPERIMENT_NAME = "simple_gan"
    cfg.__init__()                 # re-create dirs for new experiment name

    if args.epochs is not None:
        cfg.N_EPOCHS   = args.epochs
    if args.batch is not None:
        cfg.BATCH_SIZE = args.batch
    if args.no_metrics:
        cfg.COMPUTE_METRICS = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_simple_gan] Device  : {device}")
    print(f"[train_simple_gan] Epochs  : {cfg.N_EPOCHS}")
    print(f"[train_simple_gan] Latent  : {LATENT_DIM}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, test_loader = get_loaders(cfg)

    # ── Models ────────────────────────────────────────────────────────────
    gen, dis = build_simple_models(latent_dim=LATENT_DIM, device=device)

    # ── Train ─────────────────────────────────────────────────────────────
    g_losses, d_losses = train_simple_gan(
        gen, dis, train_loader, cfg, device, latent_dim=LATENT_DIM
    )
    plot_losses(g_losses, d_losses, cfg)

    # ── Metrics ───────────────────────────────────────────────────────────
    scores = {}
    if cfg.COMPUTE_METRICS:
        from src.metrics import compute_metrics_simple_gan
        scores = compute_metrics_simple_gan(
            gen, test_loader, LATENT_DIM, cfg, device
        )

    # ── Save metrics JSON ─────────────────────────────────────────────────
    payload = {
        "experiment": cfg.EXPERIMENT_NAME,
        "model":      "SimpleGAN",
        "fid":        scores.get("fid"),
        "dacid":      scores.get("dacid"),
        "g_losses":   g_losses,
        "d_losses":   d_losses,
    }
    out = cfg.RESULTS_DIR / "metrics.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[train_simple_gan] Metrics saved → {out}")

    # ── Save checkpoint ───────────────────────────────────────────────────
    ckpt_path = cfg.CHECKPOINT_DIR / "simple_gan_final.pt"
    torch.save({"gen": gen.state_dict(), "dis": dis.state_dict()}, ckpt_path)
    print(f"[train_simple_gan] Checkpoint saved → {ckpt_path}")
    print(f"\n[train_simple_gan] Done. Results in {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()