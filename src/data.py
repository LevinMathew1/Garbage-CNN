import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transform(split: str, img_size: int = 224) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class GarbageDataset(Dataset):
    def __init__(
        self,
        paths: List[str],
        labels: List[int],
        transform: transforms.Compose,
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _collect_paths_and_labels(
    data_dir: str | Path,
) -> Tuple[List[str], List[int], List[str]]:
    data_dir = Path(data_dir)

    # if the dataset ships with its own train/val/test folders, pool them all
    # so our stratified re-split uses the full dataset
    split_dirs = [data_dir / s for s in ("train", "val", "test") if (data_dir / s).is_dir()]
    source_dirs = split_dirs if split_dirs else [data_dir]

    # derive class names from whichever source dir has the most subdirs
    class_names = sorted([p.name for p in source_dirs[0].iterdir() if p.is_dir()])

    paths, labels = [], []
    for source_dir in source_dirs:
        for label_idx, class_name in enumerate(class_names):
            class_dir = source_dir / class_name
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix in IMG_EXTENSIONS:
                    paths.append(str(img_path))
                    labels.append(label_idx)

    return paths, labels, class_names


def _make_weighted_sampler(labels: List[int], num_classes: int) -> WeightedRandomSampler:
    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = torch.tensor([class_weights[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    img_size: int = 224,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 2,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    try:
        all_paths, all_labels, class_names = _collect_paths_and_labels(data_dir)
    except Exception as exc:
        raise RuntimeError(
            f"[DATA] Failed to load data from {data_dir}: {exc}\n"
            "Check the path and ensure images are in <data_dir>/<class>/*.jpg layout."
        ) from exc

    print(f"[DATA] Found {len(all_paths)} images across {len(class_names)} classes: {class_names}")

    # carve out test first, then split the rest into train/val
    rel_val = val_split / (1.0 - test_split)

    paths_tv, paths_test, labels_tv, labels_test = train_test_split(
        all_paths, all_labels,
        test_size=test_split,
        stratify=all_labels,
        random_state=SEED,
    )
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_tv, labels_tv,
        test_size=rel_val,
        stratify=labels_tv,
        random_state=SEED,
    )

    print(f"[DATA] Split — train: {len(paths_train)}, val: {len(paths_val)}, test: {len(paths_test)}")

    train_ds = GarbageDataset(paths_train, labels_train, _build_transform("train", img_size))
    val_ds   = GarbageDataset(paths_val,   labels_val,   _build_transform("val",   img_size))
    test_ds  = GarbageDataset(paths_test,  labels_test,  _build_transform("val",   img_size))

    sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        sampler = _make_weighted_sampler(labels_train, len(class_names))
        shuffle_train = False  # sampler is mutually exclusive with shuffle

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        shuffle=shuffle_train, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names
