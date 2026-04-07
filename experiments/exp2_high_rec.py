"""
experiments/exp2_high_rec.py — Experiment 2: High Reconstruction Weight.

Doubles λ_rec to 200. Hypothesis: stronger reconstruction constraint
preserves identity better but may produce less aggressive attribute edits.

Compare attribute_demo.png vs exp1 to evaluate the tradeoff.

λ_rec=200  λ_cls_D=10  λ_cls_G=1
"""

from config import Config


class Exp2Config(Config):
    # ── Identity ────────────────────────────────────────────────────────
    EXPERIMENT_NAME = "exp2_high_rec"

    # ── Loss weights ────────────────────────────────────────────────────
    LAMBDA_REC   = 200.0   # ↑ doubled — stronger identity preservation
    LAMBDA_CLS_D =  10.0
    LAMBDA_CLS_G =   1.0

    # ── Notes ────────────────────────────────────────────────────────────
    DESCRIPTION = (
        "High reconstruction weight (λ_rec=200). "
        "Hypothesis: better identity preservation, softer attribute edits."
    )
