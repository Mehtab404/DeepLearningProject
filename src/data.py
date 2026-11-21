"""Data: get the CIFAR-10 pictures (60,000 small pictures of 10 kinds of things) ready."""
import os  # work with file and folder paths
import shutil  # copy files
import time  # wait between retries

import numpy as np  # fast number lists
import torch  # main library for deep learning
from torch.utils.data import DataLoader, Subset  # DataLoader hands out images in groups; Subset keeps only some images
from torchvision import datasets, transforms  # ready-made picture datasets and picture changes

from .utils import SEED  # the fixed starting number for randomness

CIFAR_TAR = "cifar-10-python.tar.gz"  # name of the downloaded dataset file


def prepare_data(root="data", drive_dir=None, attempts=3):  # make sure the dataset is on the computer (download it if not)
    dst = os.path.join(root, CIFAR_TAR)  # where the dataset file should be on this computer
    cached = os.path.join(drive_dir, "dataset", CIFAR_TAR) if drive_dir else None  # backup copy on Google Drive (only if a Drive folder is given)
    os.makedirs(root, exist_ok=True)  # create the data folder if it is missing
    if cached and not os.path.exists(dst) and os.path.exists(cached):  # we have a Drive copy but no local copy
        shutil.copy2(cached, dst)  # copy it here (much faster than downloading)
        print("[data] restored CIFAR-10 archive from Drive cache")  # tell the user
    for i in range(1, attempts + 1):  # try the download up to 3 times
        try:  # attempt the download
            datasets.CIFAR10(root=root, train=True, download=True)  # get the 50,000 training pictures (skipped if already there)
            datasets.CIFAR10(root=root, train=False, download=True)  # get the 10,000 test pictures (skipped if already there)
            break  # worked, so stop trying
        except Exception as e:  # noqa: BLE001 - the download failed
            if i == attempts:  # this was the last try
                raise  # give up and show the error
            print(f"[retry] CIFAR-10 download failed ({type(e).__name__}); attempt {i}/{attempts}")  # tell the user it will retry
            time.sleep(10 * i)  # wait longer each time (10s, 20s, ...)
    if cached and os.path.exists(dst) and not os.path.exists(cached):  # we have a local copy but no Drive copy yet
        os.makedirs(os.path.dirname(cached), exist_ok=True)  # create the Drive folder
        shutil.copy2(dst, cached)  # save a copy to Drive for next time
        print("[data] cached CIFAR-10 archive to Drive")  # tell the user


def get_train_loader(batch_size=128, root="data", max_samples=0, augment=True, num_workers=2, seed=SEED):  # gives training pictures in shuffled groups (batches)
    tfms = []  # list of changes to apply to each picture
    if augment:  # augment = make small random changes so the model sees more variety
        tfms += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]  # slightly shift the picture; sometimes mirror it left-right
    tfms.append(transforms.ToTensor())  # turn the picture into numbers between 0 and 1 (no other rescaling here)
    ds = datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.Compose(tfms))  # the training dataset with those changes
    if max_samples and max_samples < len(ds):  # use only part of the data (for quick tests)
        g = np.random.RandomState(seed)  # random generator with a fixed start so the same pictures are picked every time
        idx = g.permutation(len(ds))[:max_samples].tolist()  # shuffle all positions and keep the first max_samples
        ds = Subset(ds, idx)  # keep only those pictures
    pin = torch.cuda.is_available()  # True if a GPU exists (then copying data to it can be faster)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,  # hand out groups of batch_size pictures in a new random order each epoch
                      pin_memory=pin, drop_last=True,  # drop_last = skip the last small group so all groups have equal size
                      persistent_workers=num_workers > 0)  # keep the helper processes alive between epochs (faster)


def stratified_indices(targets, n, seed=SEED):  # picks the same number of test pictures from every class, so the test is fair
    targets = np.asarray(targets)  # list of labels as a number array
    classes = np.unique(targets)  # the different classes: 0 to 9
    per_class = max(1, n // len(classes))  # pictures per class, e.g. 1000 // 10 = 100
    rng = np.random.RandomState(seed)  # random generator with a fixed start
    idx = []  # the chosen picture positions
    for c in classes:  # go through each class
        cls_idx = np.where(targets == c)[0]  # positions of all pictures of this class
        idx.extend(rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False).tolist())  # pick per_class of them at random, never the same one twice
    return sorted(idx)  # give back the positions in order


def get_eval_tensors(n=1000, root="data", seed=SEED):  # builds the fixed test set used to compare all models
    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())  # the test pictures as numbers between 0 and 1 (no random changes)
    idx = stratified_indices(ds.targets, n, seed)  # choose an equal number of pictures from each class
    X = torch.stack([ds[i][0] for i in idx])  # stack the chosen pictures into one big block of numbers
    Y = torch.tensor([ds.targets[i] for i in idx], dtype=torch.long)  # their true labels as whole numbers
    return X, Y  # pictures and labels
