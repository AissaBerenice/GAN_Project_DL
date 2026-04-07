"""
experiments/exp1_baseline.py — Experiment 1: Baseline AttGAN.

Uses the exact hyperparameters from the original paper.
This is the reference point for comparing experiments 2 and 3.

λ_rec=100  λ_cls_D=10  λ_cls_G=1
"""

from config import Config


class Exp1Config(Config):
    # ── Identity ────────────────────────────────────────────────────────
    EXPERIMENT_NAME = "exp1_baseline"

    # ── Loss weights — paper defaults ───────────────────────────────────
    LAMBDA_REC   = 100.0
    LAMBDA_CLS_D =  10.0
    LAMBDA_CLS_G =   1.0

    # ── Notes (for export_results.py summary) ───────────────────────────
    DESCRIPTION = (
        "Baseline AttGAN. Paper-default λ values. "
        "Balanced reconstruction vs attribute editing."
    )