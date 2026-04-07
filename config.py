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
    EXPERIMENT_NAME = "default"

    # ── Root paths ───────────────────────────────────────────────────
    ROOT = Path(__file__).parent.resolve()

    # ── CelebA dataset location ───────────────────────────────────────
    # Point this to the folder that contains:
    #   img_align_celeba/   (unzipped images)
    #   list_attr_celeba.csv
    #   list_eval_partition.csv
    #
    # LOCAL: leave as-is — put the files in  data/celeba/
    # COLAB: overridden in Cell 3 of the notebook to your Drive path
    CELEBA_DIR = ROOT / "data" / "celeba"

    @property
    def RESULTS_DIR(self):
        return self.ROOT / "results" / self.EXPERIMENT_NAME

    @property
    def CHECKPOINT_DIR(self):
        return self.ROOT / "checkpoints" / self.EXPERIMENT_NAME

    # ── Dataset ──────────────────────────────────────────────────────
    IMG_SIZE = 128

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
    LAMBDA_REC   = 100.0
    LAMBDA_CLS_D =  10.0
    LAMBDA_CLS_G =   1.0

    # ── Architecture ─────────────────────────────────────────────────
    ENC_DIM = 64
    DEC_DIM = 64
    DIS_DIM = 64

    # ── Logging ──────────────────────────────────────────────────────
    SAVE_EVERY      = 5
    LOG_EVERY_STEPS = 100

    # ── Metrics (FID + DACID) ─────────────────────────────────────────
    COMPUTE_METRICS   = True
    METRICS_N_SAMPLES = 2048

    def __init__(self):
        for d in [self.RESULTS_DIR, self.CHECKPOINT_DIR]:
            d.mkdir(parents=True, exist_ok=True)