"""CIFAR-10 data loading. Tensors stay in [0, 1] (ToTensor only); normalisation lives in the model."""
# CIFAR-10: 60,000 small colour images (32x32 pixels), 10 classes (airplane, car, bird, ...).
#   50,000 images for training, 10,000 for testing.
# Important: we do NOT normalise here. Images stay in [0, 1] so that eps (e.g. 8/255) has the same
# meaning in pixel units for every model. Normalisation is done inside the model (see models.py).
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .utils import SEED

CIFAR_TAR = "cifar-10-python.tar.gz"    # name of the downloaded archive file


def prepare_data(root="data", drive_dir=None, attempts=3):
    """Make sure CIFAR-10 is on disk, with retries and an optional Google Drive cache.

    The official host can be very slow (a Colab run measured ~60 kB/s, i.e. ~47 min for 170 MB), so the
    archive is cached in <drive_dir>/dataset/ and restored from there on later runs.
    """
    dst = os.path.join(root, CIFAR_TAR)    # where the archive should be locally
    # Optional copy on Google Drive (only used on Colab).
    cached = os.path.join(drive_dir, "dataset", CIFAR_TAR) if drive_dir else None
    os.makedirs(root, exist_ok=True)

    # 1) If a Drive copy exists and the local file does not, copy it (much faster than downloading).
    if cached and not os.path.exists(dst) and os.path.exists(cached):
        shutil.copy2(cached, dst)
        print("[data] restored CIFAR-10 archive from Drive cache")

    # 2) Download (torchvision skips it if the files are already there). Retry if the network fails,
    #    waiting longer each time (10s, 20s, ...).
    for i in range(1, attempts + 1):
        try:
            datasets.CIFAR10(root=root, train=True, download=True)
            datasets.CIFAR10(root=root, train=False, download=True)
            break    # success -> stop retrying
        except Exception as e:  # noqa: BLE001
            if i == attempts:
                raise    # last attempt failed -> give up and show the error
            print(f"[retry] CIFAR-10 download failed ({type(e).__name__}); attempt {i}/{attempts}")
            time.sleep(10 * i)

    # 3) Save a copy to Drive so the next run does not need to download again.
    if cached and os.path.exists(dst) and not os.path.exists(cached):
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        shutil.copy2(dst, cached)
        print("[data] cached CIFAR-10 archive to Drive")


def get_train_loader(batch_size=128, root="data", max_samples=0, augment=True, num_workers=2, seed=SEED):
    # Returns a DataLoader: an object that gives the training set in shuffled mini-batches of images.

    # Data augmentation = small random changes to each training image, so the model sees more variety
    # and overfits less:
    #   RandomCrop(32, padding=4): add a 4-pixel border, then cut a random 32x32 piece (small shift).
    #   RandomHorizontalFlip: mirror the image left-right with probability 0.5.
    tfms = []
    if augment:
        tfms += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    # ToTensor converts the image to a float tensor of shape (3, 32, 32) with values in [0, 1].
    tfms.append(transforms.ToTensor())  # NO Normalize here (golden rule)
    ds = datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.Compose(tfms))

    # Optionally use only a random part of the training set (for quick tests).
    if max_samples and max_samples < len(ds):
        g = np.random.RandomState(seed)    # seeded -> the same subset every run
        idx = g.permutation(len(ds))[:max_samples].tolist()
        ds = Subset(ds, idx)

    # pin_memory speeds up copying batches from CPU to GPU (only useful when a GPU exists).
    pin = torch.cuda.is_available()
    # shuffle=True: new random order every epoch.
    # drop_last=True: skip the last incomplete batch so all batches have the same size.
    # num_workers: background processes that load images in parallel.
    # persistent_workers: keep those processes alive between epochs (faster).
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=pin, drop_last=True,
                      persistent_workers=num_workers > 0)


def stratified_indices(targets, n, seed=SEED):
    """n // 10 images per class, drawn with a fixed seed. Returns a sorted index list."""
    # "Stratified" = the same number of images from every class. E.g. n=1000 -> 100 per class.
    # This makes the evaluation fair: no class is over- or under-represented.
    targets = np.asarray(targets)     # list of labels -> numpy array
    classes = np.unique(targets)      # [0, 1, ..., 9]
    per_class = max(1, n // len(classes))
    rng = np.random.RandomState(seed)
    idx = []
    for c in classes:
        cls_idx = np.where(targets == c)[0]    # positions of all images of class c
        # Pick per_class of them at random, without picking the same image twice.
        idx.extend(rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False).tolist())
    return sorted(idx)


def get_eval_tensors(n=1000, root="data", seed=SEED):
    """Fixed stratified test subset as in-memory tensors (X in [0,1], shape (N,3,32,32); Y int64)."""
    # Evaluation uses a fixed subset of the TEST set (never the training set) with no augmentation.
    # Every model is evaluated on exactly the same images, so the comparison is fair.
    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())
    idx = stratified_indices(ds.targets, n, seed)
    # Load the chosen images into one big tensor X and their labels into Y.
    X = torch.stack([ds[i][0] for i in idx])
    Y = torch.tensor([ds.targets[i] for i in idx], dtype=torch.long)
    return X, Y
