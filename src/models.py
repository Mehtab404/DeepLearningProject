"""Models: the two image-recognising models we compare (a CNN and a ViT)."""
import os  # work with file and folder paths
import time  # used to wait between retries

import torch  # main library for deep learning
import torch.nn as nn  # building blocks for models
import torch.nn.functional as F  # ready-made maths functions (we use it to resize images)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)  # average red, green, blue value of the training images
CIFAR_STD = (0.2470, 0.2435, 0.2616)  # how much red, green, blue values usually vary from that average
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # average colour of ImageNet pictures (used by the ImageNet-pretrained ResNet-18)
IMAGENET_STD = (0.229, 0.224, 0.225)  # how much ImageNet colours usually vary from that average

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",  # the 10 picture types; label 0 = airplane, 1 = automobile, ...
                   "dog", "frog", "horse", "ship", "truck"]  # ... label 9 = truck

HF_VIT_ID = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"  # name of an optional bigger ViT on the HuggingFace website
HUB_REPO = "chenyaofo/pytorch-cifar-models:master"  # GitHub place where the ready-made CNN is downloaded from


def _with_retries(fn, what="download", attempts=5, base_delay=5):  # runs fn(); if the internet fails, waits and tries again
    for i in range(1, attempts + 1):  # try up to 5 times
        try:  # attempt the job
            return fn()  # success: give back the result and stop
        except Exception as e:  # noqa: BLE001 - something went wrong (e.g. network problem)
            if i == attempts:  # this was the last try
                raise  # give up and show the error
            wait = base_delay * i  # wait longer after each failure: 5s, 10s, 15s ...
            print(f"[retry] {what} failed ({type(e).__name__}: {str(e)[:120]}); attempt {i}/{attempts}, "  # tell the user what failed
                  f"retrying in {wait}s")  # and how long we wait
            time.sleep(wait)  # pause before trying again


class NormalizedModelWrapper(nn.Module):  # a cover around a model: it resizes and rescales the image first, so outside the model images stay plain 0-to-1
    def __init__(self, model, mean=CIFAR_MEAN, std=CIFAR_STD, img_size=None):  # set up the cover
        super().__init__()  # start the basic PyTorch model setup
        self.model = model  # the real model inside (CNN or ViT)
        self.img_size = img_size  # size to enlarge images to (None = keep the size; 224 for the ViT)
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))  # store the average colour inside the model (fixed numbers, not learned)
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))  # store the colour variation inside the model (fixed numbers, not learned)

    def forward(self, x):  # runs when we give an image to the model
        if self.img_size is not None and x.shape[-1] != self.img_size:  # the ViT needs bigger images than the small 32x32 ones
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)  # enlarge the image smoothly to 224x224
        x_norm = (x - self.mean) / self.std  # subtract the average colour and divide by the variation (models learn better this way)
        out = self.model(x_norm)  # run the real model
        return out.logits if hasattr(out, 'logits') else out  # some models wrap their 10 scores in an object; take the scores out of it


def load_cnn_clean(device, pretrained=True, arch="cifar10_resnet20"):  # get the normally trained CNN (ResNet-20)
    base = _with_retries(lambda: torch.hub.load(HUB_REPO, arch, pretrained=pretrained, trust_repo=True),  # download the ready-made CNN (retry if the internet fails)
                         what="torch.hub " + arch)  # name used in the retry message
    model = NormalizedModelWrapper(base, CIFAR_MEAN, CIFAR_STD, img_size=None)  # put the cover around it; the CNN keeps 32x32 images
    return model.to(device).eval()  # move it to the GPU/CPU and switch to test mode


def load_vit_clean(device, backbone="timm-tiny", pretrained=True):  # get the ViT; also says whether it still needs normal training
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # the ViT expects colours centred at 0.5 with a spread of 0.5
    if backbone == "timm-tiny":  # option 1: small ViT (fast)
        import timm  # library that contains many ready-made models
        base = _with_retries(  # download the small ViT (retry if the internet fails)
            lambda: timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=10),  # 224x224 input cut into 16x16 pieces; 10 outputs for our 10 classes
            what="timm vit_tiny weights")  # name used in the retry message
        needs_ft = pretrained  # a downloaded ViT has a new empty 10-class end part, so it needs a short normal training
    elif backbone == "hf-base":  # option 2: bigger ViT from HuggingFace (slower)
        from transformers import AutoConfig, ViTForImageClassification  # tools to load that bigger ViT
        if pretrained:  # download the already-learned version
            base = _with_retries(lambda: ViTForImageClassification.from_pretrained(HF_VIT_ID),  # download it (retry if the internet fails)
                                 what="HF ViT-Base weights")  # name used in the retry message
        else:  # only build an empty model; saved weights will be loaded into it later
            base = ViTForImageClassification(AutoConfig.from_pretrained(HF_VIT_ID))  # same shape, random numbers
        needs_ft = False  # this one is already trained on CIFAR-10
    else:  # any other name is a mistake
        raise ValueError(f"Unknown ViT backbone: {backbone}")  # show an error
    model = NormalizedModelWrapper(base, mean, std, img_size=224)  # put the cover around it; images are enlarged to 224x224
    return model.to(device).eval(), needs_ft  # move to the GPU/CPU, test mode; also say if it needs normal training


def load_resnet18_clean(device, pretrained=True):  # get an ImageNet-pretrained ResNet-18 (about 11M weights) that takes 224x224 pictures, like the ViT
    import timm  # library that contains many ready-made models
    base = _with_retries(  # download the weights (retry if the internet fails)
        lambda: timm.create_model("resnet18", pretrained=pretrained, num_classes=10),  # new empty 10-class end part for CIFAR-10
        what="timm resnet18 weights")  # name used in the retry message
    model = NormalizedModelWrapper(base, IMAGENET_MEAN, IMAGENET_STD, img_size=224)  # same cover as the ViT: enlarge to 224x224, normalise inside the model
    return model.to(device).eval(), pretrained  # move to the GPU/CPU, test mode; a downloaded model still needs normal training


def save_checkpoint(model, path):  # save the model's learned numbers (weights) to a file
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)  # create the folder if it is missing
    torch.save(model.state_dict(), path)  # write all learned numbers to the file


def load_checkpoint(model, path, device):  # load saved weights into a model of the same shape
    model.load_state_dict(torch.load(path, map_location=device))  # read the file and copy the numbers into the model (works on GPU or CPU)
    return model.to(device).eval()  # move to the GPU/CPU and switch to test mode


def sanity_check(device=None):  # quick check that both models take a 32x32 image and give 10 scores
    device = device or torch.device("cpu")  # use the CPU if no device was given
    x = torch.rand(4, 3, 32, 32, device=device)  # 4 fake colour images of size 32x32
    cnn = load_cnn_clean(device)  # load the CNN
    vit, _ = load_vit_clean(device)  # load the ViT (ignore the second answer)
    with torch.no_grad():  # no learning information needed for a simple test (saves memory)
        for name, m in (("CNN", cnn), ("ViT", vit)):  # test both models
            out = m(x)  # get the scores for the 4 fake images
            assert out.shape == (4, 10), f"{name} output shape {tuple(out.shape)} != (4, 10)"  # must be 4 images x 10 scores, else stop with an error
            print(f"[sanity] {name}: input {tuple(x.shape)} -> output {tuple(out.shape)} OK")  # tell the user it is fine


if __name__ == "__main__":  # true only when this file is run directly
    sanity_check()  # run the quick check
