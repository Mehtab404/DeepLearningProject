"""Model definitions: normalization wrapper + pretrained CIFAR-10 CNN and ViT loaders.

Golden rule: every tensor that reaches an attack lives in [0, 1]. The (x - mean) / std step
(and bilinear upsampling for the ViT) happens INSIDE forward(), so epsilon and clamp(0, 1)
are exact in pixel space for every architecture.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

HF_VIT_ID = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"


class NormalizedModelWrapper(nn.Module):
    def __init__(self, model, mean=CIFAR_MEAN, std=CIFAR_STD, img_size=None):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        if self.img_size is not None and x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        x_norm = (x - self.mean) / self.std
        out = self.model(x_norm)
        return out.logits if hasattr(out, 'logits') else out


def load_cnn_clean(device, pretrained=True, arch="cifar10_resnet20"):
    """CNN-Clean: pretrained CIFAR-10 ResNet from chenyaofo/pytorch-cifar-models (no training needed).

    pretrained=False builds the same architecture without downloading weights (used when a local
    checkpoint will be loaded on top).
    """
    base = torch.hub.load("chenyaofo/pytorch-cifar-models", arch, pretrained=pretrained, trust_repo=True)
    model = NormalizedModelWrapper(base, CIFAR_MEAN, CIFAR_STD, img_size=None)
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
        base = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=10)
        needs_ft = pretrained  # when pretrained=False we only build the shell for a local checkpoint
    elif backbone == "hf-base":
        from transformers import AutoConfig, ViTForImageClassification
        if pretrained:
            base = ViTForImageClassification.from_pretrained(HF_VIT_ID)
        else:
            base = ViTForImageClassification(AutoConfig.from_pretrained(HF_VIT_ID))
        needs_ft = False
    else:
        raise ValueError(f"Unknown ViT backbone: {backbone}")
    model = NormalizedModelWrapper(base, mean, std, img_size=224)
    return model.to(device).eval(), needs_ft


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def sanity_check(device=None):
    """Task 2.4: both models must output (batch, 10) on a dummy [0, 1] tensor of shape (4, 3, 32, 32)."""
    device = device or torch.device("cpu")
    x = torch.rand(4, 3, 32, 32, device=device)
    cnn = load_cnn_clean(device)
    vit, _ = load_vit_clean(device)
    with torch.no_grad():
        for name, m in (("CNN", cnn), ("ViT", vit)):
            out = m(x)
            assert out.shape == (4, 10), f"{name} output shape {tuple(out.shape)} != (4, 10)"
            print(f"[sanity] {name}: input {tuple(x.shape)} -> output {tuple(out.shape)} OK")


if __name__ == "__main__":
    sanity_check()
