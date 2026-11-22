"""Data: get the CIFAR-10 pictures (60,000 small pictures of 10 kinds of things) ready."""
# WHAT: Imports the libraries used to find files, copy them, wait, handle arrays, build tensors and load
#       CIFAR-10.
# HOW: os = paths, shutil = file copy, time = sleep between retries, numpy = random sampling, torch.utils.data =
#      batching, torchvision = the CIFAR-10 dataset class and image transforms.
# WHY: Data preparation touches the file system, the network and PyTorch's data tools.
# PROS & CONS: + All standard libraries. - torchvision downloads from the University of Toronto server, which
#              can be slow (22 minutes once in Colab).
# EXAMPLE: datasets.CIFAR10(root='data', train=True, download=True) fetches 50,000 training images.
import os  # work with file and folder paths
import shutil  # copy files
import time  # wait between retries

import numpy as np  # fast number lists
import torch  # main library for deep learning
from torch.utils.data import DataLoader, Subset  # DataLoader hands out images in groups; Subset keeps only some images
from torchvision import datasets, transforms  # ready-made picture datasets and picture changes

# WHAT: Imports the project-wide random seed (42).
# HOW: Relative import from src/utils.py.
# WHY: One shared seed makes the 1000-image evaluation subset and the shuffling repeatable.
# PROS & CONS: + Single source of truth. - Changing the seed changes every result.
# EXAMPLE: SEED == 42.
from .utils import SEED  # the fixed starting number for randomness

# WHAT: File name of the CIFAR-10 archive.
# HOW: A plain string constant.
# WHY: Used both for the local copy and the optional Google Drive cache.
# PROS & CONS: + One place to change. - None.
# EXAMPLE: 'cifar-10-python.tar.gz' (about 170 MB).
CIFAR_TAR = "cifar-10-python.tar.gz"  # name of the downloaded dataset file


# WHAT: Makes sure the dataset is on disk, downloading it only if needed.
# HOW: Restore a cached archive from Drive if present, try the torchvision download up to 3 times with growing
#      waits, then cache a copy to Drive.
# WHY: Colab sessions are temporary; caching saves a long re-download after every disconnect.
# PROS & CONS: + Robust to network failures. - Needs the Drive folder to be mounted for the caching benefit.
# EXAMPLE: prepare_data('data', '/content/drive/MyDrive/adv_ckpts').
def prepare_data(root="data", drive_dir=None, attempts=3):  # make sure the dataset is on the computer (download it if not)
    # WHAT: Path where the archive should live locally.
    # HOW: Join the data folder and the archive name.
    # WHY: Used for the cache copy and the existence test.
    # PROS & CONS: + Portable across Windows/Linux. - None.
    # EXAMPLE: data/cifar-10-python.tar.gz
    dst = os.path.join(root, CIFAR_TAR)  # where the dataset file should be on this computer
    # WHAT: Path of the optional Drive backup of the archive.
    # HOW: None if no Drive folder was given.
    # WHY: Allows the same function to work with and without Drive.
    # PROS & CONS: + Optional feature. - None.
    # EXAMPLE: /content/drive/MyDrive/adv_ckpts/dataset/cifar-10-python.tar.gz
    cached = os.path.join(drive_dir, "dataset", CIFAR_TAR) if drive_dir else None  # backup copy on Google Drive (only if a Drive folder is given)
    # WHAT: Creates the data folder if it is missing.
    # HOW: exist_ok=True makes it harmless if the folder exists.
    # WHY: Copying or downloading into a missing folder would fail.
    # PROS & CONS: + Idempotent. - None.
    # EXAMPLE: A first run creates ./data.
    os.makedirs(root, exist_ok=True)  # create the data folder if it is missing
    # WHAT: Restores the archive from Drive when it is not on the local disk yet.
    # HOW: copy2 copies the file together with its timestamps.
    # WHY: Copying from Drive takes seconds; downloading takes minutes.
    # PROS & CONS: + Faster restart. - Uses Drive space (about 170 MB).
    # EXAMPLE: Prints '[data] restored CIFAR-10 archive from Drive cache'.
    if cached and not os.path.exists(dst) and os.path.exists(cached):  # we have a Drive copy but no local copy
        shutil.copy2(cached, dst)  # copy it here (much faster than downloading)
        print("[data] restored CIFAR-10 archive from Drive cache")  # tell the user
    # WHAT: Download loop with retries.
    # HOW: Each pass tries to fetch the training and test sets; on an error it waits 10*i seconds (10 s, 20 s)
    #      and tries again; the last failure is re-raised.
    # WHY: Colab network errors are usually temporary.
    # PROS & CONS: + Survives temporary outages. - Gives up after 3 tries.
    # EXAMPLE: [retry] CIFAR-10 download failed (URLError); attempt 1/3
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
    # WHAT: Saves a copy of the archive to Drive for next time.
    # HOW: Creates the Drive folder and copies the archive once.
    # WHY: Next session can use the fast restore above.
    # PROS & CONS: + One-time cost. - Requires write access to Drive.
    # EXAMPLE: Prints '[data] cached CIFAR-10 archive to Drive'.
    if cached and os.path.exists(dst) and not os.path.exists(cached):  # we have a local copy but no Drive copy yet
        os.makedirs(os.path.dirname(cached), exist_ok=True)  # create the Drive folder
        shutil.copy2(dst, cached)  # save a copy to Drive for next time
        print("[data] cached CIFAR-10 archive to Drive")  # tell the user


# WHAT: Builds the object that hands out shuffled training batches.
# HOW: Applies optional augmentation, wraps the 50,000 images in a DataLoader with batch size 128 and shuffling.
# WHY: Mini-batch training needs a stream of random batches.
# PROS & CONS: + Fast with several worker processes. - drop_last discards up to 127 images per epoch.
# EXAMPLE: get_train_loader(128) yields 390 batches per epoch.
def get_train_loader(batch_size=128, root="data", max_samples=0, augment=True, num_workers=2, seed=SEED):  # gives training pictures in shuffled groups (batches)
    # WHAT: List of image transforms to apply, built up step by step.
    # HOW: Augmentations first, then conversion to a tensor.
    # WHY: The order matters: PIL-image transforms must run before ToTensor.
    # PROS & CONS: + Flexible. - None.
    # EXAMPLE: [RandomCrop, RandomHorizontalFlip, ToTensor]
    tfms = []  # list of changes to apply to each picture
    # WHAT: Optional data augmentation.
    # HOW: RandomCrop(32, padding=4) shifts the picture by up to 4 pixels; RandomHorizontalFlip mirrors it half
    #      of the time.
    # WHY: More variety reduces overfitting. These are the standard CIFAR-10 augmentations.
    # PROS & CONS: + Cheap and effective. - Makes training accuracy lower than test-time accuracy on clean
    #              images.
    # EXAMPLE: A cat photo may be shifted 3 pixels left and flipped.
    if augment:  # augment = make small random changes so the model sees more variety
        tfms += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]  # slightly shift the picture; sometimes mirror it left-right
    # WHAT: Converts a picture to a tensor with pixel values in [0, 1].
    # HOW: PIL image 0..255 integers become floats 0..1; layout becomes (3, 32, 32).
    # WHY: The whole project keeps images in [0, 1] so that eps means the same for every model. Mean/std
    #      normalisation happens INSIDE the model wrapper.
    # PROS & CONS: + eps is exact in pixel space. - Each model must normalise internally.
    # EXAMPLE: A pixel of 128 becomes 0.502.
    tfms.append(transforms.ToTensor())  # turn the picture into numbers between 0 and 1 (no other rescaling here)
    # WHAT: Loads the 50,000 training images with the chosen transforms.
    # HOW: torchvision reads the archive and applies the transforms on the fly.
    # WHY: Standard dataset requested by the mentor.
    # PROS & CONS: + Official dataset class. - Needs the download step above.
    # EXAMPLE: len(ds) == 50000.
    ds = datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.Compose(tfms))  # the training dataset with those changes
    # WHAT: Optionally uses only part of the training set (quick tests).
    # HOW: A seeded random permutation picks max_samples positions; Subset keeps only those.
    # WHY: The CPU smoke test must finish in minutes.
    # PROS & CONS: + Deterministic subset. - Not used in full runs (max_samples = 0).
    # EXAMPLE: --max-train-samples 64 keeps 64 images.
    if max_samples and max_samples < len(ds):  # use only part of the data (for quick tests)
        g = np.random.RandomState(seed)  # random generator with a fixed start so the same pictures are picked every time
        idx = g.permutation(len(ds))[:max_samples].tolist()  # shuffle all positions and keep the first max_samples
        ds = Subset(ds, idx)  # keep only those pictures
    # WHAT: Checks if a GPU exists.
    # HOW: pin_memory speeds up CPU to GPU copies.
    # WHY: Only useful with a GPU.
    # PROS & CONS: + Faster transfer. - Uses extra pinned RAM.
    # EXAMPLE: True on a Colab T4.
    pin = torch.cuda.is_available()  # True if a GPU exists (then copying data to it can be faster)
    # WHAT: Creates the batch loader.
    # HOW: shuffle=True gives a new order each epoch; drop_last=True keeps every batch the same size;
    #      persistent_workers keeps helper processes alive.
    # WHY: Equal batch sizes make cosine schedules and BatchNorm statistics stable.
    # PROS & CONS: + Fast and stable. - Loses a few images per epoch.
    # EXAMPLE: batch_size=128, num_workers=2.
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,  # hand out groups of batch_size pictures in a new random order each epoch
                      pin_memory=pin, drop_last=True,  # drop_last = skip the last small group so all groups have equal size
                      persistent_workers=num_workers > 0)  # keep the helper processes alive between epochs (faster)


# WHAT: Picks the same number of test images from every class.
# HOW: For each of the 10 classes choose n/10 images at random (seeded, no repeats) and return the sorted
#      positions.
# WHY: A balanced subset avoids a class imbalance that could bias accuracy; the same subset is used for every
#      model so results are comparable.
# PROS & CONS: + Fair and repeatable. - A subset of 1000 has a standard error of up to 1.6 points.
# EXAMPLE: n = 1000 gives 100 images per class.
def stratified_indices(targets, n, seed=SEED):  # picks the same number of test pictures from every class, so the test is fair
    # WHAT: Gets the class labels and the list of classes.
    # HOW: Converts the label list to a numpy array and finds the unique classes.
    # WHY: Needed to sample per class.
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: classes = [0, 1, ..., 9].
    targets = np.asarray(targets)  # list of labels as a number array
    classes = np.unique(targets)  # the different classes: 0 to 9
    # WHAT: Number of images per class.
    # HOW: Integer division n // number of classes, at least 1.
    # WHY: Keeps the classes balanced.
    # PROS & CONS: + Guarantees balance. - n not divisible by 10 loses a few images.
    # EXAMPLE: 1000 // 10 = 100.
    per_class = max(1, n // len(classes))  # pictures per class, e.g. 1000 // 10 = 100
    # WHAT: Seeded random generator.
    # HOW: RandomState(seed) always produces the same sequence.
    # WHY: Makes the subset identical in every session.
    # PROS & CONS: + Reproducible. - Legacy numpy API (still fine).
    # EXAMPLE: Seed 42 picks the same 1000 images every time.
    rng = np.random.RandomState(seed)  # random generator with a fixed start
    # WHAT: Collects the chosen image positions.
    # HOW: Starts empty and grows per class.
    # WHY: Needed to build the final subset.
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: []
    idx = []  # the chosen picture positions
    # WHAT: Samples within each class.
    # HOW: np.where finds all positions of class c; rng.choice picks per_class without replacement.
    # WHY: Sampling without replacement avoids duplicates.
    # PROS & CONS: + Balanced. - Python loop over only 10 classes is cheap.
    # EXAMPLE: 100 images drawn from the 1000 'cat' test images.
    for c in classes:  # go through each class
        cls_idx = np.where(targets == c)[0]  # positions of all pictures of this class
        idx.extend(rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False).tolist())  # pick per_class of them at random, never the same one twice
    # WHAT: Returns positions in increasing order.
    # HOW: Sorting makes the order deterministic.
    # WHY: Same tensor layout in every run.
    # PROS & CONS: + Deterministic. - None.
    # EXAMPLE: [3, 17, 24, ...]
    return sorted(idx)  # give back the positions in order


# WHAT: Builds the fixed evaluation tensors X (images) and Y (labels).
# HOW: Loads the 10,000 test images with ToTensor only (no augmentation), selects the stratified subset, stacks
#      it into one tensor.
# WHY: Attacks are applied batch-wise on this one fixed block, so every model is tested on identical images.
# PROS & CONS: + Fair comparison, fast repeated evaluation. - The subset (N = 1000) is smaller than the full
#              10,000 test set.
# EXAMPLE: X.shape == (1000, 3, 32, 32), Y.shape == (1000,).
def get_eval_tensors(n=1000, root="data", seed=SEED):  # builds the fixed test set used to compare all models
    # WHAT: Loads the test set without random augmentation.
    # HOW: train=False selects the 10,000 test images.
    # WHY: Test images must not be randomly altered.
    # PROS & CONS: + Clean evaluation. - None.
    # EXAMPLE: len(ds) == 10000.
    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())  # the test pictures as numbers between 0 and 1 (no random changes)
    # WHAT: Chooses which test images to use.
    # HOW: Calls stratified_indices with the labels.
    # WHY: Gives the balanced subset.
    # PROS & CONS: + Balanced. - None.
    # EXAMPLE: 1000 indices.
    idx = stratified_indices(ds.targets, n, seed)  # choose an equal number of pictures from each class
    # WHAT: Stacks the chosen images into one tensor.
    # HOW: ds[i][0] is the i-th image; stack adds a batch dimension.
    # WHY: One tensor is faster to move to the GPU and to batch.
    # PROS & CONS: + Fast. - 12 MB of RAM for 1000 images (about 123 MB for all 10,000).
    # EXAMPLE: torch.Size([1000, 3, 32, 32]).
    X = torch.stack([ds[i][0] for i in idx])  # stack the chosen pictures into one big block of numbers
    # WHAT: Labels of the chosen images as integers.
    # HOW: dtype long is what cross-entropy expects.
    # WHY: Matches the model output indices.
    # PROS & CONS: + Correct dtype. - None.
    # EXAMPLE: tensor([0, 0, ..., 9]).
    Y = torch.tensor([ds.targets[i] for i in idx], dtype=torch.long)  # their true labels as whole numbers
    # WHAT: Returns images and labels.
    # HOW: Tuple of two tensors.
    # WHY: Used by evaluation and plotting.
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: X, Y = get_eval_tensors(1000).
    return X, Y  # pictures and labels
