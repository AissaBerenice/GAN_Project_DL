"""
src/dataset.py — CelebA dataset wrapper for AttGAN.

Loads the CelebA dataset via torchvision, filters the 40 default labels
down to the attributes defined in Config, and normalises them to {-1, +1}.

Usage:
    from src.dataset import get_loaders
    train_loader, test_loader = get_loaders(cfg)
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CelebA


class CelebAAttrDataset(Dataset):
    """
    Thin wrapper around torchvision.datasets.CelebA that:
      • crops and resizes images to cfg.IMG_SIZE × cfg.IMG_SIZE
      • selects only the attribute columns listed in cfg.ATTRS
      • converts binary {0, 1} labels → {-1, +1} for bipolar conditioning
    """

    def __init__(self, root, split: str, attr_names: list[str],
                 img_size: int, download: bool = True):
        """
        Args:
            root       : directory where CelebA will be downloaded / cached
            split      : 'train' | 'valid' | 'test'
            attr_names : list of attribute name strings (subset of CelebA's 40)
            img_size   : spatial size after resize (square)
            download   : download dataset if not already present
        """
        transform = transforms.Compose([
            transforms.CenterCrop(178),           # remove excess chin / forehead
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),    # mild data augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # → [−1, 1]
        ])

        self._ds = CelebA(
            root=str(root),
            split=split,
            target_type="attr",
            transform=transform,
            download=download,
        )

        all_names   = self._ds.attr_names
        self._idx   = [all_names.index(a) for a in attr_names]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i):
        img, attrs = self._ds[i]
        sel = attrs[self._idx].float()   # shape (n_attrs,),  values {0, 1}
        sel = sel * 2 - 1                # → {-1, +1}
        return img, sel


# ──────────────────────────────────────────────────────────────────────
def get_loaders(cfg) -> tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders from Config.

    Returns:
        (train_loader, test_loader)
    """
    kw = dict(
        attr_names=cfg.ATTRS,
        img_size=cfg.IMG_SIZE,
    )

    train_ds = CelebAAttrDataset(cfg.DATA_DIR, "train", **kw)
    test_ds  = CelebAAttrDataset(cfg.DATA_DIR, "test",  **kw)

    loader_kw = dict(
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        **loader_kw,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        **loader_kw,
    )

    print(f"[dataset] Train: {len(train_ds):,} samples | "
          f"Test: {len(test_ds):,} samples")
    return train_loader, test_loader