"""
train.py — Main entry point for AttGAN training.

Run from the repo root:
    python train.py
    python train.py --exp exp1_baseline
    python train.py --exp exp2_high_rec
    python train.py --exp exp3_strong_attr
    python train.py --exp exp1_baseline --resume checkpoints/exp1_baseline/ckpt_epoch010.pt
    python train.py --exp exp1_baseline --eval-only --resume checkpoints/exp1_baseline/ckpt_epoch030.pt
    python train.py --no-metrics
"""

import argparse
import importlib
import torch

from config      import Config
from src.dataset import get_loaders
from src.models  import build_models
from src.trainer import Trainer
from src.utils   import (plot_losses, attribute_demo,
                          evaluate_attribute_accuracy,
                          evaluate_reconstruction)

_EXP_MAP = {
    "exp1_baseline":    "experiments.exp1_baseline.Exp1Config",
    "exp2_high_rec":    "experiments.exp2_high_rec.Exp2Config",
    "exp3_strong_attr": "experiments.exp3_low_rec.Exp3Config",
}


def _load_config(exp_name):
    if exp_name is None:
        return Config()
    module_path, class_name = _EXP_MAP[exp_name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def parse_args():
    p = argparse.ArgumentParser(description="Train AttGAN on CelebA")
    p.add_argument("--exp",        type=str, default=None,
                   choices=list(_EXP_MAP.keys()))
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch",      type=int, default=None)
    p.add_argument("--resume",     type=str, default=None)
    p.add_argument("--eval-only",  action="store_true")
    p.add_argument("--no-metrics", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = _load_config(args.exp)

    if args.epochs    is not None: cfg.N_EPOCHS        = args.epochs
    if args.batch     is not None: cfg.BATCH_SIZE       = args.batch
    if args.no_metrics:            cfg.COMPUTE_METRICS  = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Experiment : {cfg.EXPERIMENT_NAME}")
    print(f"[train] Device     : {device}")
    if device.type == "cuda":
        print(f"[train] GPU        : {torch.cuda.get_device_name(0)}")
    print(f"[train] Epochs     : {cfg.N_EPOCHS}  Batch: {cfg.BATCH_SIZE}")
    print(f"[train] λ_rec={cfg.LAMBDA_REC}  λ_cls_D={cfg.LAMBDA_CLS_D}  λ_cls_G={cfg.LAMBDA_CLS_G}")
    print(f"[train] Metrics    : {cfg.COMPUTE_METRICS}\n")

    train_loader, test_loader = get_loaders(cfg)
    enc, gen, dis = build_models(cfg, device)

    trainer = Trainer(enc, gen, dis, train_loader, test_loader, cfg, device)

    if not args.eval_only:
        g_losses, d_losses = trainer.train(resume_path=args.resume)
        plot_losses(g_losses, d_losses, cfg)
    else:
        if args.resume is None:
            raise ValueError("--eval-only requires --resume <path>")
        from src.utils import load_checkpoint
        load_checkpoint(args.resume, enc, gen, dis)

    print("\n[train] Running qualitative evaluation...")
    evaluate_attribute_accuracy(enc, gen, dis, test_loader, cfg)
    evaluate_reconstruction(enc, gen, test_loader, cfg)
    attribute_demo(enc, gen, test_loader, cfg)

    print(f"\n[train] Done. Results → {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()