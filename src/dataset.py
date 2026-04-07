"""
src/dataset.py — CelebA dataset loader for AttGAN (Kaggle file format).

Reads directly from the files provided in the Kaggle CelebA dataset
(https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
without relying on torchvision's broken Google Drive downloader.

Expected directory layout (set cfg.CELEBA_DIR to this folder):

    <CELEBA_DIR>/
        img_align_celeba/          <- unzipped from img_align_celeba.zip
            000001.jpg
            000002.jpg
            ...
        list_attr_celeba.csv       <- attribute labels  (values: 1 / -1)
        list_eval_partition.csv    <- train/val/test split  (values: 0 / 1 / 2)

Splits match torchvision's defaults exactly:
    partition 0 -> 'train'   (162,770 images)
    partition 1 -> 'valid'   ( 19,867 images)
    partition 2 -> 'test'    ( 19,962 images)

Attribute values in the CSV are already {-1, +1} — no conversion needed.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


_SPLIT_MAP = {"train": 0, "valid": 1, "test": 2}

ALL_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]


class CelebAAttrDataset(Dataset):
    """
    CelebA dataset loaded from Kaggle CSV files.

    Returns (image_tensor, attr_tensor) where:
        image_tensor : FloatTensor (3, img_size, img_size)  in [-1, 1]
        attr_tensor  : FloatTensor (n_attrs,)               in {-1, +1}
    """

    def __init__(self, celeba_dir, split: str,
                 attr_names: list, img_size: int):

        celeba_dir = Path(celeba_dir)
        self._img_dir = celeba_dir / "img_align_celeba"
        self._check_files(celeba_dir)

        # Load and merge partition + attribute CSVs
        part_df = pd.read_csv(celeba_dir / "list_eval_partition.csv")
        attr_df = pd.read_csv(celeba_dir / "list_attr_celeba.csv")
        part_df.columns = part_df.columns.str.strip()
        attr_df.columns = attr_df.columns.str.strip()

        merged   = part_df.merge(attr_df, on="image_id")
        split_df = merged[merged["partition"] == _SPLIT_MAP[split]].reset_index(drop=True)

        self._filenames  = split_df["image_id"].tolist()
        self._attr_array = split_df[attr_names].values.astype("float32")
        # CSV values are already {-1, +1} — no conversion needed

        self._transform = T.Compose([
            T.CenterCrop(178),
            T.Resize(img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

    @staticmethod
    def _check_files(celeba_dir: Path):
        needed = [
            "img_align_celeba",
            "list_attr_celeba.csv",
            "list_eval_partition.csv",
        ]
        missing = [n for n in needed if not (celeba_dir / n).exists()]
        if missing:
            raise FileNotFoundError(
                f"\nMissing CelebA files in: {celeba_dir}\n"
                + "\n".join(f"  MISSING: {m}" for m in missing)
                + "\n\nSee notebook Cell 3 for upload instructions."
            )

    def __len__(self) -> int:
        return len(self._filenames)

    def __getitem__(self, idx: int):
        img   = Image.open(self._img_dir / self._filenames[idx]).convert("RGB")
        img   = self._transform(img)
        attrs = torch.from_numpy(self._attr_array[idx])
        return img, attrs


def get_loaders(cfg):
    """
    Build train and test DataLoaders from Config.
    Reads from cfg.CELEBA_DIR — see notebook Cell 3 for how to set this.
    Returns: (train_loader, test_loader)
    """
    kw = dict(celeba_dir=cfg.CELEBA_DIR, attr_names=cfg.ATTRS, img_size=cfg.IMG_SIZE)

    train_ds = CelebAAttrDataset(split="train", **kw)
    test_ds  = CelebAAttrDataset(split="test",  **kw)

    loader_kw = dict(num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, **loader_kw)

    print(f"[dataset] train={len(train_ds):,}  test={len(test_ds):,}")
    print(f"[dataset] source: {cfg.CELEBA_DIR}")
    return train_loader, test_loader