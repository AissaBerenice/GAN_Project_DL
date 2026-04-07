"""
config.py — Central configuration for AttGAN + Simple GAN.

All hyperparameters and directory paths live here.

    from config import Config
    cfg = Config()

For experiments, subclass Config and override only what changes:
    from experiments.exp1_baseline import Exp1Config
"""

from pathlib import Path


class Config:
    # ── Experiment identity ───────────────────────────────────────────
    # Each experiment saves results to results/<EXPERIMENT_NAME>/
    # and checkpoints to checkpoints/<EXPERIMENT_NAME>/
    EXPERIMENT_NAME = "default"

    # ── Root paths ───────────────────────────────────────────────────
    ROOT     = Path(__file__).parent.resolve()
    DATA_DIR = ROOT / "data"          # torchvision downloads CelebA here

    @property
    def RESULTS_DIR(self):
        return self.ROOT / "results" / self.EXPERIMENT_NAME

    @property
    def CHECKPOINT_DIR(self):
        return self.ROOT / "checkpoints" / self.EXPERIMENT_NAME

    # ── Dataset ──────────────────────────────────────────────────────
    IMG_SIZE = 128

    # 13 attributes selected from CelebA's 40 binary labels
    ATTRS = [
        "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair",
        "Bushy_Eyebrows", "Eyeglasses", "Male",
        "Mouth_Slightly_Open", "Mustache", "No_Beard",
        "Pale_Skin", "Young",
    ]
    N_ATTRS = len(ATTRS)   # 13

    # ── Training ─────────────────────────────────────────────────────
    BATCH_SIZE  = 32
    N_EPOCHS    = 30
    LR          = 0.0002
    BETA1       = 0.5
    BETA2       = 0.999
    NUM_WORKERS = 2

    # ── Loss weights (AttGAN paper defaults) ─────────────────────────
    LAMBDA_REC   = 100.0   # reconstruction fidelity
    LAMBDA_CLS_D =  10.0   # discriminator attribute classification
    LAMBDA_CLS_G =   1.0   # generator attribute classification

    # ── Architecture depth ────────────────────────────────────────────
    ENC_DIM = 64   # base channels for Encoder
    DEC_DIM = 64   # base channels for Generator / Decoder
    DIS_DIM = 64   # base channels for Discriminator

    # ── Logging ──────────────────────────────────────────────────────
    SAVE_EVERY      = 5    # save samples + checkpoint every N epochs
    LOG_EVERY_STEPS = 100  # print loss every N batches

    # ── Metrics (FID + DACID) ─────────────────────────────────────────
    # Set True to compute FID + DACID after training (~5 extra min on T4)
    COMPUTE_METRICS   = True
    METRICS_N_SAMPLES = 2048

    def __init__(self):
        """Create output directories on instantiation."""
        for d in [self.DATA_DIR, self.RESULTS_DIR, self.CHECKPOINT_DIR]:
            d.mkdir(parents=True, exist_ok=True)
