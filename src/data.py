"""CIFAR-10 data loading. Tensors stay in [0, 1] (ToTensor only); normalisation lives in the model."""
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .utils import SEED

CIFAR_TAR = "cifar-10-python.tar.gz"


def prepare_data(root="data", drive_dir=None, attempts=3):
    """Make sure CIFAR-10 is on disk, with retries and an optional Google Drive cache.

    The official host can be very slow (a Colab run measured ~60 kB/s, i.e. ~47 min for 170 MB), so the
    archive is cached in <drive_dir>/dataset/ and restored from there on later runs.
    """
    dst = os.path.join(root, CIFAR_TAR)
    cached = os.path.join(drive_dir, "dataset", CIFAR_TAR) if drive_dir else None
    os.makedirs(root, exist_ok=True)
    if cached and not os.path.exists(dst) and os.path.exists(cached):
        shutil.copy2(cached, dst)
        print("[data] restored CIFAR-10 archive from Drive cache")
    for i in range(1, attempts + 1):
        try:
            datasets.CIFAR10(root=root, train=True, download=True)
            datasets.CIFAR10(root=root, train=False, download=True)
            break
        except Exception as e:  # noqa: BLE001
            if i == attempts:
                raise
            print(f"[retry] CIFAR-10 download failed ({type(e).__name__}); attempt {i}/{attempts}")
            time.sleep(10 * i)
    if cached and os.path.exists(dst) and not os.path.exists(cached):
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        shutil.copy2(dst, cached)
        print("[data] cached CIFAR-10 archive to Drive")


def get_train_loader(batch_size=128, root="data", max_samples=0, augment=True, num_workers=2, seed=SEED):
    tfms = []
    if augment:
        tfms += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    tfms.append(transforms.ToTensor())  # NO Normalize here (golden rule)
    ds = datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.Compose(tfms))
    if max_samples and max_samples < len(ds):
        g = np.random.RandomState(seed)
        idx = g.permutation(len(ds))[:max_samples].tolist()
        ds = Subset(ds, idx)
    pin = torch.cuda.is_available()
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=pin, drop_last=True,
                      persistent_workers=num_workers > 0)


def stratified_indices(targets, n, seed=SEED):
    """n // 10 images per class, drawn with a fixed seed. Returns a sorted index list."""
    targets = np.asarray(targets)
    classes = np.unique(targets)
    per_class = max(1, n // len(classes))
    rng = np.random.RandomState(seed)
    idx = []
    for c in classes:
        cls_idx = np.where(targets == c)[0]
        idx.extend(rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False).tolist())
    return sorted(idx)


def get_eval_tensors(n=1000, root="data", seed=SEED):
    """Fixed stratified test subset as in-memory tensors (X in [0,1], shape (N,3,32,32); Y int64)."""
    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())
    idx = stratified_indices(ds.targets, n, seed)
    X = torch.stack([ds[i][0] for i in idx])
    Y = torch.tensor([ds.targets[i] for i in idx], dtype=torch.long)
    return X, Y
