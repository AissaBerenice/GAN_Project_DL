"""
experiments/exp3_low_rec.py — Experiment 3: Stronger Attribute Signal.

Halves λ_rec to 50 and raises λ_cls_G to 5. Hypothesis: the generator
is pushed harder to satisfy the requested attributes, at the cost of
some identity fidelity. Expect more visible edits but possible artefacts.

λ_rec=50  λ_cls_D=10  λ_cls_G=5
"""

from config import Config


class Exp3Config(Config):
    # ── Identity ────────────────────────────────────────────────────────
    EXPERIMENT_NAME = "exp3_strong_attr"

    # ── Loss weights ────────────────────────────────────────────────────
    LAMBDA_REC   =  50.0   # ↓ halved — less identity constraint
    LAMBDA_CLS_D =  10.0
    LAMBDA_CLS_G =   5.0   # ↑ stronger attribute signal to G

    # ── Notes ────────────────────────────────────────────────────────────
    DESCRIPTION = (
        "Strong attribute signal (λ_rec=50, λ_cls_G=5). "
        "Hypothesis: sharper attribute edits, reduced identity preservation."
    )
