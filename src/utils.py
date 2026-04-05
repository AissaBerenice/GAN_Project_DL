"""
src/utils.py — Visualisation, checkpointing, and evaluation utilities.

Functions
---------
denorm              : inverse-normalise a tensor for display
visualise_samples   : save/show an epoch-progress image grid
plot_losses         : plot G and D loss curves
save_checkpoint     : persist model state dicts
load_checkpoint     : restore model state dicts and return epoch
attribute_demo      : per-image, toggle each attribute independently
evaluate_attribute_accuracy : measure classification accuracy on generated images
evaluate_reconstruction     : measure L1 reconstruction error
"""

import os
import torch
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────

def denorm(x: torch.Tensor) -> torch.Tensor:
    """Map values from [−1, 1] → [0, 1] for matplotlib / torchvision."""
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def tensor_to_np(t: torch.Tensor):
    """(C, H, W) tensor → (H, W, C) numpy array in [0, 1]."""
    return denorm(t).cpu().permute(1, 2, 0).numpy()


# ─────────────────────────────────────────────────────────────────────
# Sample visualisation
# ─────────────────────────────────────────────────────────────────────

def visualise_samples(enc, gen, test_imgs, test_attrs, epoch: int, cfg):
    """
    Save and display a comparison grid for one fixed test batch.

    Rows:
        0  — original images
        1  — reconstructed (same attributes)
        2+ — first 4 attributes each flipped independently

    Args:
        enc, gen    : Encoder and Generator (eval mode set internally)
        test_imgs   : (B, 3, H, W) tensor on device
        test_attrs  : (B, n_attrs) tensor on device
        epoch       : current epoch number (used in filename)
        cfg         : Config instance
    """
    enc.eval(); gen.eval()
    device = test_imgs.device
    n = test_imgs.size(0)

    with torch.no_grad():
        z   = enc(test_imgs)
        rec = gen(z, test_attrs)

        edits = []
        for attr_i in range(min(4, cfg.N_ATTRS)):
            a_flip = test_attrs.clone()
            a_flip[:, attr_i] = -a_flip[:, attr_i]
            edits.append(gen(z, a_flip))

    rows   = [test_imgs.cpu(), rec.cpu()] + [e.cpu() for e in edits]
    labels = ["Original", "Reconstruct"] + \
             [f"Flip: {cfg.ATTRS[i]}" for i in range(len(edits))]

    fig, axes = plt.subplots(len(labels), n,
                              figsize=(n * 2.0, len(labels) * 2.2))
    for r, (row_imgs, lbl) in enumerate(zip(rows, labels)):
        for c in range(n):
            ax = axes[r][c]
            ax.imshow(tensor_to_np(row_imgs[c]))
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(lbl, fontsize=7, rotation=0,
                               labelpad=72, va="center")

    plt.suptitle(f"Epoch {epoch}", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out = cfg.RESULTS_DIR / f"samples_epoch{epoch:03d}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"[utils] Saved → {out}")

    enc.train(); gen.train()


# ─────────────────────────────────────────────────────────────────────
# Loss curves
# ─────────────────────────────────────────────────────────────────────

def plot_losses(g_losses: list[float], d_losses: list[float], cfg):
    """Plot and save Generator / Discriminator loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(g_losses, color="#e74c3c", label="G loss")
    ax1.set_title("Generator loss"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(d_losses, color="#2980b9", label="D loss")
    ax2.set_title("Discriminator loss"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Training dynamics", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = cfg.RESULTS_DIR / "loss_curves.png"
    plt.savefig(out, dpi=100)
    plt.show()
    print(f"[utils] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────

def save_checkpoint(enc, gen, dis, epoch: int, cfg):
    """Save model state dicts to checkpoints/ckpt_epoch<N>.pt"""
    path = cfg.CHECKPOINT_DIR / f"ckpt_epoch{epoch:03d}.pt"
    torch.save({
        "epoch": epoch,
        "enc": enc.state_dict(),
        "gen": gen.state_dict(),
        "dis": dis.state_dict(),
    }, path)
    print(f"[utils] Checkpoint saved → {path}")


def load_checkpoint(path: str, enc, gen, dis) -> int:
    """
    Load model state dicts from a checkpoint file.

    Returns:
        epoch number stored in the checkpoint
    """
    ckpt = torch.load(path, map_location="cpu")
    enc.load_state_dict(ckpt["enc"])
    gen.load_state_dict(ckpt["gen"])
    dis.load_state_dict(ckpt["dis"])
    epoch = ckpt.get("epoch", 0)
    print(f"[utils] Resumed from epoch {epoch} ({path})")
    return epoch


# ─────────────────────────────────────────────────────────────────────
# Attribute manipulation demo
# ─────────────────────────────────────────────────────────────────────

def attribute_demo(enc, gen, test_loader, cfg, n_imgs: int = 4):
    """
    For n_imgs test images, show the original alongside every attribute
    toggled independently — one column per attribute.

    Args:
        enc, gen    : trained Encoder and Generator
        test_loader : DataLoader for test split
        cfg         : Config instance
        n_imgs      : number of sample images to include (rows)
    """
    enc.eval(); gen.eval()
    imgs, attrs = next(iter(test_loader))
    imgs  = imgs[:n_imgs]
    attrs = attrs[:n_imgs]
    device = next(enc.parameters()).device
    imgs  = imgs.to(device)
    attrs = attrs.to(device)

    with torch.no_grad():
        z = enc(imgs)
        n_cols = cfg.N_ATTRS + 1   # original + one per attribute

        fig, axes = plt.subplots(n_imgs, n_cols,
                                  figsize=(n_cols * 1.9, n_imgs * 2.1))

        for img_i in range(n_imgs):
            # Column 0 — original
            axes[img_i][0].imshow(tensor_to_np(imgs[img_i]))
            axes[img_i][0].axis("off")
            if img_i == 0:
                axes[img_i][0].set_title("Original", fontsize=6,
                                          fontweight="bold")

            # Columns 1..n_attrs — each attribute flipped
            for attr_i, attr_name in enumerate(cfg.ATTRS):
                a_edit = attrs[img_i : img_i + 1].clone()
                cur = a_edit[0, attr_i].item()
                a_edit[0, attr_i] = -cur
                edited = gen(z[img_i : img_i + 1], a_edit)

                ax = axes[img_i][attr_i + 1]
                ax.imshow(tensor_to_np(edited[0]))
                ax.axis("off")
                if img_i == 0:
                    sign = "+" if cur < 0 else "−"
                    ax.set_title(f"{sign}{attr_name}", fontsize=5.5)

    plt.suptitle(
        "Attribute manipulation demo\n"
        "(each column = one attribute flipped from its current value)",
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout()
    out = cfg.RESULTS_DIR / "attribute_demo.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"[utils] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────
# Quantitative evaluation
# ─────────────────────────────────────────────────────────────────────

def evaluate_attribute_accuracy(enc, gen, dis, test_loader, cfg,
                                  n_batches: int = 20) -> float:
    """
    Estimate how accurately the generator satisfies target attributes.

    Uses the discriminator's cls_head to predict which attributes appear
    in generated images, then compares with the target attribute vector.

    Returns:
        overall accuracy (float, 0–100)
    """
    enc.eval(); gen.eval(); dis.eval()
    device  = next(enc.parameters()).device
    correct = torch.zeros(cfg.N_ATTRS, device=device)
    total   = 0

    with torch.no_grad():
        for i, (imgs, attrs) in enumerate(test_loader):
            if i >= n_batches:
                break
            imgs  = imgs.to(device)
            attrs = attrs.to(device)
            B     = imgs.size(0)

            perm         = torch.randperm(B)
            target_attrs = attrs[perm]

            fakes      = gen(enc(imgs), target_attrs)
            _, cls     = dis(fakes)
            pred       = (cls > 0).float() * 2 - 1   # {0,1} → {-1,+1}
            correct   += (pred == target_attrs).sum(dim=0)
            total     += B

    acc_per = (correct / total * 100).cpu().numpy()
    overall  = float(acc_per.mean())

    print("\n[eval] Attribute accuracy on translated images:")
    for name, acc in zip(cfg.ATTRS, acc_per):
        bar = "█" * int(acc / 5)
        print(f"  {name:<22} {acc:5.1f}%  {bar}")
    print(f"\n  Overall: {overall:.1f}%")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(cfg.ATTRS, acc_per, color="#3498db", edgecolor="white")
    ax.axhline(overall, color="#e74c3c", ls="--",
               label=f"Mean {overall:.1f}%")
    ax.set_ylim(0, 105)
    ax.set_xticklabels(cfg.ATTRS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Attribute classification accuracy on generated images",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = cfg.RESULTS_DIR / "attr_accuracy.png"
    plt.savefig(out, dpi=100)
    plt.show()
    print(f"[eval] Saved → {out}")

    enc.train(); gen.train(); dis.train()
    return overall


def evaluate_reconstruction(enc, gen, test_loader, cfg,
                              n_batches: int = 20) -> float:
    """
    Compute average L1 reconstruction error on the test set.

    Returns:
        average L1 loss (float) — lower is better
    """
    from torch.nn import L1Loss
    enc.eval(); gen.eval()
    device  = next(enc.parameters()).device
    l1      = L1Loss()
    total, n = 0.0, 0

    with torch.no_grad():
        for i, (imgs, attrs) in enumerate(test_loader):
            if i >= n_batches:
                break
            imgs  = imgs.to(device)
            attrs = attrs.to(device)
            rec   = gen(enc(imgs), attrs)
            total += l1(rec, imgs).item()
            n     += 1

    avg = total / n
    print(f"\n[eval] Reconstruction L1 (test, {n} batches): {avg:.4f}")
    print("       Lower = better identity preservation")

    enc.train(); gen.train()
    return avg