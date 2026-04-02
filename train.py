"""
train.py — Main entry point for AttGAN training.

Run from the repo root:
    python train.py
    python train.py --epochs 10
    python train.py --resume checkpoints/ckpt_epoch010.pt

All hyperparameters are configured in config.py.
"""

import argparse
import torch

from config      import Config
from src.dataset import get_loaders
from src.models  import build_models
from src.trainer import Trainer
from src.utils   import plot_losses, attribute_demo, \
                         evaluate_attribute_accuracy, evaluate_reconstruction


def parse_args():
    p = argparse.ArgumentParser(description="Train AttGAN on CelebA")
    p.add_argument("--epochs",  type=int,   default=None,
                   help="Override Config.N_EPOCHS")
    p.add_argument("--batch",   type=int,   default=None,
                   help="Override Config.BATCH_SIZE")
    p.add_argument("--resume",  type=str,   default=None,
                   help="Path to checkpoint .pt file to resume training")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, run evaluation only (requires --resume)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()

    # Allow CLI overrides
    if args.epochs is not None:
        cfg.N_EPOCHS    = args.epochs
    if args.batch is not None:
        cfg.BATCH_SIZE  = args.batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device  : {device}")
    if device.type == "cuda":
        print(f"[train] GPU     : {torch.cuda.get_device_name(0)}")

    print(f"[train] Epochs  : {cfg.N_EPOCHS}")
    print(f"[train] Batch   : {cfg.BATCH_SIZE}")
    print(f"[train] Attrs   : {cfg.ATTRS}\n")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, test_loader = get_loaders(cfg)

    # ── Models ────────────────────────────────────────────────────────
    enc, gen, dis = build_models(cfg, device)

    # ── Train ─────────────────────────────────────────────────────────
    trainer = Trainer(enc, gen, dis, train_loader, test_loader, cfg, device)

    if not args.eval_only:
        g_losses, d_losses = trainer.train(resume_path=args.resume)
        plot_losses(g_losses, d_losses, cfg)
    else:
        if args.resume is None:
            raise ValueError("--eval-only requires --resume <checkpoint>")
        from src.utils import load_checkpoint
        load_checkpoint(args.resume, enc, gen, dis)

    # ── Evaluate ──────────────────────────────────────────────────────
    print("\n[train] Running evaluation...")
    evaluate_attribute_accuracy(enc, gen, dis, test_loader, cfg)
    evaluate_reconstruction(enc, gen, test_loader, cfg)
    attribute_demo(enc, gen, test_loader, cfg)

    print("\n[train] All done. Results saved to:", cfg.RESULTS_DIR)


if __name__ == "__main__":
    main()