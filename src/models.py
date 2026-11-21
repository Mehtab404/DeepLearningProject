"""Model definitions: normalization wrapper + pretrained CIFAR-10 CNN and ViT loaders.

Golden rule: every tensor that reaches an attack lives in [0, 1]. The (x - mean) / std step
(and bilinear upsampling for the ViT) happens INSIDE forward(), so epsilon and clamp(0, 1)
are exact in pixel space for every architecture.
"""
# Two model families are compared:
#   CNN (Convolutional Neural Network): ResNet-20. It looks at small local patches with sliding filters.
#   ViT (Vision Transformer): splits the image into 16x16 patches and uses self-attention, so every
#       patch can look at every other patch (global view).
# Both are PRETRAINED (someone already trained them), so we only need to fine-tune them.
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================================================================================================
# EASY-LANGUAGE SUMMARY OF THIS FILE (read this first)
# ---------------------------------------------------------------------------------------------------
# What is this file?   It creates / loads / saves the two models we compare.
#   CNN = ResNet-20.  Looks at the picture through small sliding windows (filters), step by step.
#   ViT = Vision Transformer (ViT-Tiny). Cuts the picture into square patches (like puzzle pieces)
#         and lets every patch "talk to" every other patch (this is called self-attention).
# Both come PRE-TRAINED (already learned), so we do not train from zero.
# Key trick: NormalizedModelWrapper. The model itself does the resizing + normalising, so outside the
#            model every picture is a plain 0-to-1 picture. That keeps "eps" fair for both models.
# Functions here: load_cnn_clean, load_vit_clean, save_checkpoint, load_checkpoint, sanity_check.
# "checkpoint" = a saved file of the model's learned weights (.pth).
# ===================================================================================================

# Per-channel (R, G, B) mean and standard deviation of the CIFAR-10 training images.
# The pretrained ResNet was trained on images normalised with these numbers.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Label i means CIFAR10_CLASSES[i], e.g. label 3 = "cat".
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

# Where the pretrained weights come from:
HF_VIT_ID = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"    # HuggingFace ViT-Base (optional)
HUB_REPO = "chenyaofo/pytorch-cifar-models:master"                    # GitHub repo with CIFAR ResNets


def _with_retries(fn, what="download", attempts=5, base_delay=5):
    """Run fn(); retry transient network failures (HTTP 5xx, timeouts) with a growing delay."""
    # Downloads sometimes fail for a moment. Instead of crashing, try again up to `attempts` times,
    # waiting 5s, 10s, 15s, ... between tries.
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - urllib/requests/hub raise many different error types
            if i == attempts:
                raise
            wait = base_delay * i
            print(f"[retry] {what} failed ({type(e).__name__}: {str(e)[:120]}); attempt {i}/{attempts}, "
                  f"retrying in {wait}s")
            time.sleep(wait)


class NormalizedModelWrapper(nn.Module):
    """Wraps a network so it accepts raw [0,1] images: resize (ViT only) -> normalise -> network -> logits.

    This is why epsilon is measured in true pixel units for both CNN and ViT."""
    # Why a wrapper? The attacks add noise to the RAW image and clip it to [0, 1]. If normalisation were
    # done in the data loader, eps would mean something different for each model. Putting resize +
    # normalisation inside the model keeps the attack code identical and fair for CNN and ViT.
    def __init__(self, model, mean=CIFAR_MEAN, std=CIFAR_STD, img_size=None):
        super().__init__()
        self.model = model            # the real network (ResNet or ViT)
        self.img_size = img_size      # None for the CNN (keeps 32x32), 224 for the ViT
        # register_buffer: stores mean/std inside the model (moved to GPU with .to(device), saved in the
        # checkpoint) but they are NOT trainable parameters.
        # view(1, 3, 1, 1) reshapes them so they broadcast over (batch, channels, height, width).
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        # The ViT was pretrained on 224x224 images, so the 32x32 CIFAR image is enlarged with bilinear
        # interpolation. This step is differentiable, so attack gradients still flow back to the 32x32 pixels.
        if self.img_size is not None and x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        # Normalisation is part of the model, so attacks never see normalised tensors.
        x_norm = (x - self.mean) / self.std
        # EASY: normalising = subtract the average colour, divide by the spread. Models learn better this way.
        out = self.model(x_norm)
        # HuggingFace models return an object with .logits; torchvision/timm return a tensor.
        return out.logits if hasattr(out, 'logits') else out


def load_cnn_clean(device, pretrained=True, arch="cifar10_resnet20"):
    """CNN-Clean: pretrained CIFAR-10 ResNet from chenyaofo/pytorch-cifar-models (no training needed).

    pretrained=False builds the same architecture without downloading weights (used when a local
    checkpoint will be loaded on top).
    """
    # "Clean" = trained normally on clean images only (no adversarial training yet). This is the baseline.
    # ":master" pins the branch. Without it torch.hub first probes github.com/<repo>/tree/main/ to guess
    # the default branch, and a transient GitHub 5xx on that probe crashed a full Colab run.
    base = _with_retries(lambda: torch.hub.load(HUB_REPO, arch, pretrained=pretrained, trust_repo=True),
                         what="torch.hub " + arch)
    model = NormalizedModelWrapper(base, CIFAR_MEAN, CIFAR_STD, img_size=None)
    # .eval() = evaluation mode (BatchNorm uses stored statistics, dropout is off).
    return model.to(device).eval()


def load_vit_clean(device, backbone="timm-tiny", pretrained=True):
    """ViT-Clean.

    backbone="timm-tiny": timm vit_tiny_patch16_224 (ImageNet weights, new 10-class head).
        Returns needs_clean_finetune=True -> caller runs a short clean fine-tune (default 5 epochs).
    backbone="hf-base":   HuggingFace ViT-Base already fine-tuned on CIFAR-10. No clean training,
        but ~16x more compute per image than ViT-Tiny, so adversarial fine-tuning is much slower.

    Returns (model, needs_clean_finetune).
    """
    # Both ViTs expect 224x224 input normalised with mean=std=0.5.
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if backbone == "timm-tiny":
        import timm
        # "patch16_224": 224x224 input cut into 16x16 patches -> 14x14 = 196 patch tokens.
        # num_classes=10 replaces the original 1000-class ImageNet head with a new, untrained 10-class
        # head. That new head is why a short clean fine-tune on CIFAR-10 is needed.
        base = _with_retries(
            lambda: timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=10),
            what="timm vit_tiny weights")
        needs_ft = pretrained  # when pretrained=False we only build the shell for a local checkpoint
    elif backbone == "hf-base":
        from transformers import AutoConfig, ViTForImageClassification
        if pretrained:
            base = _with_retries(lambda: ViTForImageClassification.from_pretrained(HF_VIT_ID),
                                 what="HF ViT-Base weights")
        else:
            # Build the architecture only (random weights); a saved checkpoint is loaded on top later.
            base = ViTForImageClassification(AutoConfig.from_pretrained(HF_VIT_ID))
        needs_ft = False
    else:
        raise ValueError(f"Unknown ViT backbone: {backbone}")
    model = NormalizedModelWrapper(base, mean, std, img_size=224)
    return model.to(device).eval(), needs_ft


def save_checkpoint(model, path):
    # state_dict() = a dictionary of all weights (and buffers). Saving it lets us reuse a trained model.
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    # Load saved weights into a model with the same architecture.
    # map_location lets a GPU-trained checkpoint be loaded on a CPU-only machine (and vice versa).
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def sanity_check(device=None):
    """Task 2.4: both models must output (batch, 10) on a dummy [0, 1] tensor of shape (4, 3, 32, 32)."""
    # Check that both models accept 32x32 CIFAR-sized input and return 10 class scores per image.
    device = device or torch.device("cpu")
    x = torch.rand(4, 3, 32, 32, device=device)
    cnn = load_cnn_clean(device)
    vit, _ = load_vit_clean(device)
    with torch.no_grad():    # no gradients needed for a simple forward pass (saves memory)
        for name, m in (("CNN", cnn), ("ViT", vit)):
            out = m(x)
            assert out.shape == (4, 10), f"{name} output shape {tuple(out.shape)} != (4, 10)"
            print(f"[sanity] {name}: input {tuple(x.shape)} -> output {tuple(out.shape)} OK")


if __name__ == "__main__":
    sanity_check()
