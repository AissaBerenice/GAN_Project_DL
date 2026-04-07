"""
src/metrics.py — Image quality metrics: FID and DACID.

Two metrics are implemented:

  FID  (Fréchet Inception Distance)
    Standard GAN evaluation metric (Heusel et al., 2017).
    Computes the Fréchet distance between multivariate Gaussians
    fitted to Inception-v3 feature vectors of real and fake images.
    Lower = better (0 would mean identical distributions).

  DACID  (Dany Aissa & Clara's Image Distance)
    A lighter alternative based on the L2 distance between the
    *mean* Inception feature vectors of real and fake batches.
    Unlike FID it ignores covariance, making it faster and
    more interpretable for quick iteration comparisons.
    Lower = better.

Both metrics use the same Inception-v3 feature extractor
(pool_3 layer, 2048-dim), so they can be computed in one pass.

Usage:
    from src.metrics import compute_metrics
    scores = compute_metrics(enc, gen, test_loader, cfg, device)
    # → {"fid": 45.2, "dacid": 12.8}
"""

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from scipy.linalg import sqrtm


# ─────────────────────────────────────────────────────────────────────
# Inception feature extractor
# ─────────────────────────────────────────────────────────────────────

def _build_inception(device: torch.device):
    """
    Load Inception-v3 pre-trained on ImageNet and hook the pool_3
    (2048-dim) layer so we get feature vectors instead of class logits.
    """
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT,
        transform_input=False,   # we do our own normalisation
    ).to(device)
    inception.eval()

    # Replace the final classifier with Identity so forward() → pool features
    inception.fc = torch.nn.Identity()
    return inception


_INCEPTION_MEAN = [0.485, 0.456, 0.406]
_INCEPTION_STD  = [0.229, 0.224, 0.225]

_preprocess = T.Compose([
    T.Resize((299, 299), antialias=True),
    T.Normalize(_INCEPTION_MEAN, _INCEPTION_STD),
])


def _preprocess_batch(imgs: torch.Tensor) -> torch.Tensor:
    """
    Prepare a batch of images for Inception:
      • Repeat grayscale to 3 channels if needed
      • Resize to 299×299
      • Normalise with ImageNet statistics
    Images are expected in [−1, 1] (as output by our generators).
    """
    imgs = (imgs.clamp(-1, 1) + 1) / 2          # → [0, 1]
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)           # grayscale → RGB
    return torch.stack([_preprocess(img) for img in imgs])


@torch.no_grad()
def _extract_features(inception, images: torch.Tensor,
                       batch_size: int = 64) -> np.ndarray:
    """
    Run images through Inception in mini-batches and return
    the pool_3 feature matrix of shape (N, 2048).
    """
    device = next(inception.parameters()).device
    feats  = []

    for start in range(0, len(images), batch_size):
        batch = _preprocess_batch(images[start : start + batch_size]).to(device)
        out   = inception(batch)
        feats.append(out.cpu().numpy())

    return np.concatenate(feats, axis=0)   # (N, 2048)


# ─────────────────────────────────────────────────────────────────────
# DACID score
# ─────────────────────────────────────────────────────────────────────

def dacid_score(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """
    DACID — Dany Aissa & Clara's Image Distance.

    Computes the L2 distance between the centroid (mean feature vector)
    of the real distribution and that of the fake distribution.

    Intuition: if the generator captures the same "average" content as
    the real data in feature space, the centroids coincide (score → 0).
    This is faster than FID because it skips covariance estimation.

    Args:
        real_feats : (N, 2048) Inception features for real images
        fake_feats : (N, 2048) Inception features for generated images

    Returns:
        DACID score (float) — lower is better
    """
    mu_real = np.mean(real_feats, axis=0)
    mu_fake = np.mean(fake_feats, axis=0)
    return float(np.linalg.norm(mu_real - mu_fake))


# ─────────────────────────────────────────────────────────────────────
# FID score
# ─────────────────────────────────────────────────────────────────────

def fid_score(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """
    Fréchet Inception Distance (Heusel et al., 2017).

    Fits a multivariate Gaussian to each set of features and computes
    the Fréchet distance between them:

        FID = ||μ_r − μ_f||² + Tr(Σ_r + Σ_f − 2·√(Σ_r·Σ_f))

    Args:
        real_feats : (N, 2048) Inception features for real images
        fake_feats : (N, 2048) Inception features for generated images

    Returns:
        FID score (float) — lower is better
    """
    mu_r  = np.mean(real_feats, axis=0)
    mu_f  = np.mean(fake_feats, axis=0)
    cov_r = np.cov(real_feats, rowvar=False)
    cov_f = np.cov(fake_feats, rowvar=False)

    diff  = mu_r - mu_f
    # Matrix square root via scipy — add small eps for numerical stability
    covmean, _ = sqrtm(cov_r @ cov_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(cov_r + cov_f - 2 * covmean))
    return fid


# ─────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(enc, gen, test_loader, cfg, device: torch.device) -> dict:
    """
    Compute FID and DACID for an AttGAN model.

    Generates cfg.METRICS_N_SAMPLES fake images (by translating real
    images with shuffled target attributes) and compares them to an
    equal number of real images using Inception features.

    Args:
        enc, gen    : trained Encoder and Generator
        test_loader : DataLoader for the test split
        cfg         : Config instance
        device      : torch.device

    Returns:
        dict with keys "fid" and "dacid"
    """
    print("\n[metrics] Building Inception feature extractor...")
    inception = _build_inception(device)

    enc.eval(); gen.eval()
    real_list, fake_list = [], []
    collected = 0

    for imgs, attrs in test_loader:
        if collected >= cfg.METRICS_N_SAMPLES:
            break
        imgs  = imgs.to(device)
        attrs = attrs.to(device)
        B     = imgs.size(0)

        perm   = torch.randperm(B)
        target = attrs[perm]

        z     = enc(imgs)
        fakes = gen(z, target)

        real_list.append(imgs.cpu())
        fake_list.append(fakes.cpu())
        collected += B

    real_imgs = torch.cat(real_list)[:cfg.METRICS_N_SAMPLES]
    fake_imgs = torch.cat(fake_list)[:cfg.METRICS_N_SAMPLES]

    print(f"[metrics] Extracting Inception features "
          f"({len(real_imgs)} images each)...")
    real_feats = _extract_features(inception, real_imgs)
    fake_feats = _extract_features(inception, fake_imgs)

    fid   = fid_score(real_feats, fake_feats)
    dacid = dacid_score(real_feats, fake_feats)

    print(f"[metrics] FID   : {fid:.4f}  (lower = better)")
    print(f"[metrics] DACID : {dacid:.4f}  (lower = better)")

    enc.train(); gen.train()
    return {"fid": round(fid, 4), "dacid": round(dacid, 4)}


# ─────────────────────────────────────────────────────────────────────
# Standalone version for Simple GAN (no Encoder)
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics_simple_gan(gen, test_loader, latent_dim: int,
                                cfg, device: torch.device) -> dict:
    """
    Compute FID and DACID for a simple (unconditional) GAN.

    Args:
        gen        : trained Generator
        test_loader: DataLoader (real images)
        latent_dim : noise vector dimension
        cfg        : Config instance
        device     : torch.device

    Returns:
        dict with keys "fid" and "dacid"
    """
    print("\n[metrics] Building Inception feature extractor...")
    inception = _build_inception(device)

    gen.eval()
    real_list, fake_list = [], []
    collected = 0

    for imgs, _ in test_loader:
        if collected >= cfg.METRICS_N_SAMPLES:
            break
        B   = imgs.size(0)
        z   = torch.randn(B, latent_dim, 1, 1, device=device)
        fakes = gen(z)

        real_list.append(imgs.cpu())
        fake_list.append(fakes.cpu())
        collected += B

    real_imgs = torch.cat(real_list)[:cfg.METRICS_N_SAMPLES]
    fake_imgs = torch.cat(fake_list)[:cfg.METRICS_N_SAMPLES]

    print(f"[metrics] Extracting Inception features "
          f"({len(real_imgs)} images each)...")
    real_feats = _extract_features(inception, real_imgs)
    fake_feats = _extract_features(inception, fake_imgs)

    fid   = fid_score(real_feats, fake_feats)
    dacid = dacid_score(real_feats, fake_feats)

    print(f"[metrics] FID   : {fid:.4f}")
    print(f"[metrics] DACID : {dacid:.4f}")

    gen.train()
    return {"fid": round(fid, 4), "dacid": round(dacid, 4)}