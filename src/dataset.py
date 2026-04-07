"""
src/dataset.py — CelebA dataset wrapper for AttGAN.

Downloads and loads CelebA via torchvision. On first run torchvision
will attempt to download from Google Drive (~1.4 GB). If Google Drive
quota is exceeded run the fallback cell in the notebook.

Splits used (identical to torchvision defaults):
    'train'  — 162,770 images
    'valid'  —  19,867 images
    'test'   —  19,962 images

Usage:
    from src.dataset import get_loaders
    train_loader, test_loader = get_loaders(cfg)
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import CelebA


class CelebAAttrDataset(Dataset):
    """
    Thin wrapper around torchvision.datasets.CelebA that:
      • crops + resizes images to cfg.IMG_SIZE x cfg.IMG_SIZE
      • selects only the attribute columns listed in cfg.ATTRS
      • converts {0, 1} labels -> {-1, +1} for bipolar conditioning

    Args:
        root       : directory where CelebA will be downloaded / cached
        split      : 'train' | 'valid' | 'test'
        attr_names : list of attribute name strings (subset of CelebA's 40)
        img_size   : spatial size after crop+resize (square)
        download   : attempt to download if not already present
    """

    def __init__(self, root, split: str, attr_names: list,
                 img_size: int, download: bool = True):

        transform = T.Compose([
            T.CenterCrop(178),              # remove chin / forehead padding
            T.Resize(img_size),
            T.RandomHorizontalFlip(),       # mild data augmentation
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

        self._ds = CelebA(
            root=str(root),
            split=split,
            target_type="attr",
            transform=transform,
            download=download,
        )

        all_names  = self._ds.attr_names          # list of 40 strings
        self._idx  = [all_names.index(a) for a in attr_names]

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        img, attrs = self._ds[idx]
        sel = attrs[self._idx].float()   # (n_attrs,)  values {0, 1}
        sel = sel * 2 - 1                # -> {-1, +1}
        return img, sel


def get_loaders(cfg):
    """
    Build train and test DataLoaders from Config.

    Downloads CelebA to cfg.DATA_DIR on first call.

    Returns:
        (train_loader, test_loader)
    """
    kw = dict(
        root=cfg.DATA_DIR,
        attr_names=cfg.ATTRS,
        img_size=cfg.IMG_SIZE,
        download=True,
    )

    train_ds = CelebAAttrDataset(split="train", **kw)
    test_ds  = CelebAAttrDataset(split="test",  **kw)

    loader_kw = dict(
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE,
        shuffle=True,  **loader_kw,
    )
    test_loader = DataLoader(
        test_ds,  batch_size=cfg.BATCH_SIZE,
        shuffle=False, **loader_kw,
    )

    print(f"[dataset] Train : {len(train_ds):,} samples")
    print(f"[dataset] Test  : {len(test_ds):,} samples")
    print(f"[dataset] Attrs : {cfg.ATTRS}")
    return train_loader, test_loader
