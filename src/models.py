"""Models: the two image-recognising models we compare (a CNN and a ViT)."""
# WHAT: Imports file/time helpers, PyTorch, its layer module and its functional API.
# HOW: os for checkpoint folders, time.sleep for retries, nn.Module as the base class for the wrapper,
#      F.interpolate for resizing.
# WHY: Everything needed to build, wrap, save and load the CNN and the ViT.
# PROS & CONS: + Standard. - None.
# EXAMPLE: F.interpolate(x, size=(224, 224), mode='bilinear') enlarges a 32x32 image.
import os  # work with file and folder paths
import time  # used to wait between retries

import torch  # main library for deep learning
import torch.nn as nn  # building blocks for models
import torch.nn.functional as F  # ready-made maths functions (we use it to resize images)

# WHAT: Per-channel mean and standard deviation of the CIFAR-10 training images.
# HOW: Tuples of three numbers (red, green, blue).
# WHY: The pretrained ResNet-20 was trained on images normalised with exactly these values.
# PROS & CONS: + Matches the pretrained weights. - Wrong values would silently lower accuracy.
# EXAMPLE: (0.4914, 0.4822, 0.4465) is the mean colour of CIFAR-10.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)  # average red, green, blue value of the training images
CIFAR_STD = (0.2470, 0.2435, 0.2616)  # how much red, green, blue values usually vary from that average
# WHAT: Mean and standard deviation of ImageNet images.
# HOW: Standard torchvision/timm constants.
# WHY: The ImageNet-pretrained ResNet-18 expects this normalisation.
# PROS & CONS: + Matches pretraining. - Only used by the optional ResNet-18 experiment.
# EXAMPLE: (0.485, 0.456, 0.406).
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # average colour of ImageNet pictures (used by the ImageNet-pretrained ResNet-18)
IMAGENET_STD = (0.229, 0.224, 0.225)  # how much ImageNet colours usually vary from that average

# WHAT: Names of the 10 classes in label order.
# HOW: A list; position = label number.
# WHY: Used to print readable predictions in the example figure.
# PROS & CONS: + Readable output. - Must match the dataset order.
# EXAMPLE: CIFAR10_CLASSES[6] == 'frog'.
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",  # the 10 picture types; label 0 = airplane, 1 = automobile, ...
                   "dog", "frog", "horse", "ship", "truck"]  # ... label 9 = truck

# WHAT: Names of optional external model sources.
# HOW: A HuggingFace model id and the GitHub repo that holds the pretrained ResNet-20.
# WHY: Single place to change the sources.
# PROS & CONS: + Easy to update. - The GitHub download needs internet.
# EXAMPLE: 'chenyaofo/pytorch-cifar-models:master'.
HF_VIT_ID = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"  # name of an optional bigger ViT on the HuggingFace website
HUB_REPO = "chenyaofo/pytorch-cifar-models:master"  # GitHub place where the ready-made CNN is downloaded from


# WHAT: Runs a function and retries it if the network fails.
# HOW: Up to 5 attempts, waiting 5 s, 10 s, 15 s ... between them; the last failure is re-raised.
# WHY: Model downloads from GitHub/HuggingFace sometimes fail on Colab.
# PROS & CONS: + Fewer lost runs. - A permanent error still wastes about 1 minute of waiting.
# EXAMPLE: [retry] torch.hub cifar10_resnet20 failed ...; attempt 1/5, retrying in 5s
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


# WHAT: A cover around any classifier so that images outside the model stay plain pixels in [0, 1].
# HOW: forward() optionally resizes, then subtracts the mean and divides by the std, then calls the real model
#      and returns its 10 scores.
# WHY: Attacks add eps directly to pixels. If normalisation lived outside the model, eps would mean different
#      things for different models. Inside the model, eps/255 is exact for CNN and ViT alike.
# PROS & CONS: + Fair, exact epsilon for every architecture. - Every forward pass repeats the normalisation
#              (tiny cost).
# EXAMPLE: wrapper(torch.rand(4, 3, 32, 32)) returns scores of shape (4, 10).
class NormalizedModelWrapper(nn.Module):  # a cover around a model: it resizes and rescales the image first, so outside the model images stay plain 0-to-1
    # WHAT: Stores the real model and the normalisation constants.
    # HOW: register_buffer saves mean and std inside the module without making them learnable.
    # WHY: Buffers move to the GPU with the model and are saved in checkpoints.
    # PROS & CONS: + Travel with the model. - Included in state_dict (harmless).
    # EXAMPLE: wrapper.mean.shape == (1, 3, 1, 1).
    def __init__(self, model, mean=CIFAR_MEAN, std=CIFAR_STD, img_size=None):  # set up the cover
        super().__init__()  # start the basic PyTorch model setup
        self.model = model  # the real model inside (CNN or ViT)
        self.img_size = img_size  # size to enlarge images to (None = keep the size; 224 for the ViT)
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))  # store the average colour inside the model (fixed numbers, not learned)
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))  # store the colour variation inside the model (fixed numbers, not learned)

    # WHAT: Defines what happens when an image batch is given to the model.
    # HOW: Resize (ViT only), normalise, run the model, return logits.
    # WHY: This is the single entry point that attacks and training use.
    # PROS & CONS: + Clean interface. - None.
    # EXAMPLE: scores = wrapper(x).
    def forward(self, x):  # runs when we give an image to the model
        # WHAT: Enlarges small images for the ViT.
        # HOW: Bilinear interpolation from 32x32 to 224x224, only if the size differs.
        # WHY: ViT-Tiny/16 expects 224x224 input (14x14 patches of 16x16 pixels).
        # PROS & CONS: + Lets a 32x32 dataset use an ImageNet model; the attack still perturbs 32x32 pixels. -
        #              49x more pixels to process, so the ViT is much slower than the CNN; the upsampling may
        #              also smooth the perturbation.
        # EXAMPLE: (4, 3, 32, 32) becomes (4, 3, 224, 224).
        if self.img_size is not None and x.shape[-1] != self.img_size:  # the ViT needs bigger images than the small 32x32 ones
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)  # enlarge the image smoothly to 224x224
        # WHAT: Normalises the image.
        # HOW: (x - mean) / std per colour channel.
        # WHY: Networks learn faster and predict better with centred inputs.
        # PROS & CONS: + Matches the pretrained weights. - None.
        # EXAMPLE: A pixel 0.5 with mean 0.5, std 0.5 becomes 0.0 (ViT).
        x_norm = (x - self.mean) / self.std  # subtract the average colour and divide by the variation (models learn better this way)
        # WHAT: Calls the real model.
        # HOW: Forward pass on the normalised image.
        # WHY: Produces the class scores.
        # PROS & CONS: + Direct. - None.
        # EXAMPLE: out has 10 scores per image.
        out = self.model(x_norm)  # run the real model
        # WHAT: Returns only the scores.
        # HOW: HuggingFace models wrap them in an object with a .logits field; plain models return a tensor.
        # WHY: Lets both model families share the same interface.
        # PROS & CONS: + Works for both. - Uses a runtime attribute check.
        # EXAMPLE: out.logits.shape == (batch, 10).
        return out.logits if hasattr(out, 'logits') else out  # some models wrap their 10 scores in an object; take the scores out of it


# WHAT: Loads the ResNet-20 for CIFAR-10.
# HOW: torch.hub downloads the architecture (and optionally pretrained weights) from
#      chenyaofo/pytorch-cifar-models, then wraps it with CIFAR normalisation.
# WHY: ResNet-20 is a standard small CIFAR CNN with 0.27M weights; public weights give a strong baseline
#      immediately.
# PROS & CONS: + No training needed (91.5% clean accuracy). - You did not train it yourself; trained models can
#              be loaded with pretrained=False plus a checkpoint.
# EXAMPLE: load_cnn_clean(device, pretrained=False) builds the architecture with random weights.
def load_cnn_clean(device, pretrained=True, arch="cifar10_resnet20"):  # get the normally trained CNN (ResNet-20)
    # WHAT: Downloads the model definition and weights.
    # HOW: torch.hub.load fetches GitHub code; trust_repo=True skips the interactive question.
    # WHY: Single line to get a known-good model.
    # PROS & CONS: + Convenient. - Depends on a third-party repository staying online.
    # EXAMPLE: ~170 MB download is cached under ~/.cache/torch/hub.
    base = _with_retries(lambda: torch.hub.load(HUB_REPO, arch, pretrained=pretrained, trust_repo=True),  # download the ready-made CNN (retry if the internet fails)
                         what="torch.hub " + arch)  # name used in the retry message
    # WHAT: Wraps the CNN so inputs stay in [0, 1].
    # HOW: img_size=None means no resizing.
    # WHY: CNN keeps native 32x32 resolution.
    # PROS & CONS: + Fast. - None.
    # EXAMPLE: model(x) with x in [0, 1].
    model = NormalizedModelWrapper(base, CIFAR_MEAN, CIFAR_STD, img_size=None)  # put the cover around it; the CNN keeps 32x32 images
    # WHAT: Moves the CNN to the GPU/CPU and sets test mode.
    # HOW: eval() freezes BatchNorm statistics and disables dropout.
    # WHY: Attacks and evaluation must see deterministic predictions.
    # PROS & CONS: + Deterministic. - Must call train() again for training.
    # EXAMPLE: model.training is False.
    return model.to(device).eval()  # move it to the GPU/CPU and switch to test mode


# WHAT: Loads a Vision Transformer and says whether it still needs clean fine-tuning.
# HOW: Builds ViT-Tiny/16 from timm (5.7M weights, ImageNet pretrained, new 10-class head) or optionally
#      ViT-Base from HuggingFace.
# WHY: ViT-Tiny is small enough for a free Colab GPU and comparable in spirit to a small CNN.
# PROS & CONS: + Fast to train compared with ViT-Base. - The new head is random, so it must be fine-tuned on
#              CIFAR-10.
# EXAMPLE: model, needs_ft = load_vit_clean(device)  # needs_ft is True for a downloaded ViT.
def load_vit_clean(device, backbone="timm-tiny", pretrained=True):  # get the ViT; also says whether it still needs normal training
    # WHAT: Normalisation constants for ViTs.
    # HOW: Centre at 0.5 with scale 0.5, mapping [0, 1] to [-1, 1].
    # WHY: ViT-Tiny from timm was trained with this normalisation.
    # PROS & CONS: + Matches pretraining. - Different from the CNN's constants (handled inside each wrapper).
    # EXAMPLE: Pixel 1.0 maps to 1.0, pixel 0.0 to -1.0.
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # the ViT expects colours centred at 0.5 with a spread of 0.5
    # WHAT: Option 1: the small ViT.
    # HOW: timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10).
    # WHY: Small enough for a T4 (74.7 minutes of adversarial training).
    # PROS & CONS: + Fast. - Lower capacity than ViT-Base.
    # EXAMPLE: 5.7M parameters.
    if backbone == "timm-tiny":  # option 1: small ViT (fast)
        import timm  # library that contains many ready-made models
        base = _with_retries(  # download the small ViT (retry if the internet fails)
            lambda: timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=10),  # 224x224 input cut into 16x16 pieces; 10 outputs for our 10 classes
            what="timm vit_tiny weights")  # name used in the retry message
        needs_ft = pretrained  # a downloaded ViT has a new empty 10-class end part, so it needs a short normal training
    # WHAT: Option 2: the big ViT-Base already trained on CIFAR-10.
    # HOW: Loaded from HuggingFace; no extra fine-tuning needed.
    # WHY: Offered as a stronger alternative.
    # PROS & CONS: + Higher accuracy. - Much slower; weights exceed GitHub's size limit.
    # EXAMPLE: Not used for the reported results.
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
    # WHAT: Wraps the ViT with the 224x224 resize.
    # HOW: img_size=224 triggers bilinear upsampling inside forward().
    # WHY: ViT-Tiny/16 requires 224x224 input.
    # PROS & CONS: + Same [0, 1] interface as the CNN. - Slower.
    # EXAMPLE: (4, 3, 32, 32) input -> (4, 10) scores.
    model = NormalizedModelWrapper(base, mean, std, img_size=224)  # put the cover around it; images are enlarged to 224x224
    # WHAT: Returns the model and the 'needs fine-tuning' flag.
    # HOW: Tuple of (model, bool).
    # WHY: main.py trains the ViT only when the flag is True.
    # PROS & CONS: + Explicit. - Callers must unpack two values.
    # EXAMPLE: vit, needs_ft = load_vit_clean(device).
    return model.to(device).eval(), needs_ft  # move to the GPU/CPU, test mode; also say if it needs normal training


# WHAT: Optional comparison model: an ImageNet-pretrained ResNet-18 with the same 224x224 input as the ViT.
# HOW: timm resnet18 with a new 10-class head, wrapped with ImageNet normalisation and bilinear upsampling to
#      224.
# WHY: A size- and input-matched CNN would separate 'architecture' from 'size/pretraining' in the CNN-vs-ViT
#      comparison.
# PROS & CONS: + Fairer comparison with the ViT. - 11M weights and 224x224 inputs make it much slower than
#              ResNet-20; not used in the submitted report.
# EXAMPLE: scripts/extra_cnn_experiments.py --only resnet18.
def load_resnet18_clean(device, pretrained=True):  # get an ImageNet-pretrained ResNet-18 (about 11M weights) that takes 224x224 pictures, like the ViT
    import timm  # library that contains many ready-made models
    base = _with_retries(  # download the weights (retry if the internet fails)
        lambda: timm.create_model("resnet18", pretrained=pretrained, num_classes=10),  # new empty 10-class end part for CIFAR-10
        what="timm resnet18 weights")  # name used in the retry message
    model = NormalizedModelWrapper(base, IMAGENET_MEAN, IMAGENET_STD, img_size=224)  # same cover as the ViT: enlarge to 224x224, normalise inside the model
    # WHAT: Returns the ResNet-18 and whether it still needs fine-tuning.
    # HOW: A downloaded pretrained model has a new random head, so the flag equals `pretrained`.
    # WHY: Same convention as load_vit_clean.
    # PROS & CONS: + Consistent. - None.
    # EXAMPLE: (model, True).
    return model.to(device).eval(), pretrained  # move to the GPU/CPU, test mode; a downloaded model still needs normal training


# WHAT: Saves a model's learned numbers to a file.
# HOW: torch.save(state_dict) after creating the folder.
# WHY: Training takes up to 75 minutes; checkpoints prevent losing it.
# PROS & CONS: + Small files (22 MB ViT-Tiny, 1 MB ResNet-20). - Saves weights only, not the architecture.
# EXAMPLE: save_checkpoint(model, 'checkpoints/cnn_adv.pth').
def save_checkpoint(model, path):  # save the model's learned numbers (weights) to a file
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)  # create the folder if it is missing
    torch.save(model.state_dict(), path)  # write all learned numbers to the file


# WHAT: Loads saved weights into a model of the same shape.
# HOW: torch.load with map_location, load_state_dict, then eval().
# WHY: Lets any session reuse trained models.
# PROS & CONS: + Works on GPU or CPU. - The architecture must match exactly.
# EXAMPLE: load_checkpoint(load_cnn_clean(device, pretrained=False), path, device).
def load_checkpoint(model, path, device):  # load saved weights into a model of the same shape
    model.load_state_dict(torch.load(path, map_location=device))  # read the file and copy the numbers into the model (works on GPU or CPU)
    return model.to(device).eval()  # move to the GPU/CPU and switch to test mode


# WHAT: Quick check that both models accept a 32x32 batch and output 10 scores.
# HOW: Runs a random batch through the CNN and the ViT and asserts the output shape.
# WHY: Catches wiring mistakes before a long run.
# PROS & CONS: + Cheap. - Does not test accuracy.
# EXAMPLE: '[sanity] CNN: input (4, 3, 32, 32) -> output (4, 10) OK'.
def sanity_check(device=None):  # quick check that both models take a 32x32 image and give 10 scores
    device = device or torch.device("cpu")  # use the CPU if no device was given
    # WHAT: Creates 4 random fake images.
    # HOW: Uniform values in [0, 1).
    # WHY: No dataset needed for this test.
    # PROS & CONS: + Offline. - Random noise only.
    # EXAMPLE: Shape (4, 3, 32, 32).
    x = torch.rand(4, 3, 32, 32, device=device)  # 4 fake colour images of size 32x32
    # WHAT: Loads both models for the test.
    # HOW: Downloads weights if they are not cached.
    # WHY: Both must work.
    # PROS & CONS: + Tests the real loaders. - Needs internet on first use.
    # EXAMPLE: cnn, vit.
    cnn = load_cnn_clean(device)  # load the CNN
    vit, _ = load_vit_clean(device)  # load the ViT (ignore the second answer)
    # WHAT: Disables gradient tracking for the check.
    # HOW: No computation graph is built.
    # WHY: Saves memory; no learning is needed.
    # PROS & CONS: + Faster. - None.
    # EXAMPLE: Memory use is about half.
    with torch.no_grad():  # no learning information needed for a simple test (saves memory)
        for name, m in (("CNN", cnn), ("ViT", vit)):  # test both models
            out = m(x)  # get the scores for the 4 fake images
            # WHAT: The actual check.
            # HOW: Output must have exactly 4 rows and 10 columns.
            # WHY: Wrong shapes would break the loss and the attacks.
            # PROS & CONS: + Clear failure message. - None.
            # EXAMPLE: (4, 10) passes.
            assert out.shape == (4, 10), f"{name} output shape {tuple(out.shape)} != (4, 10)"  # must be 4 images x 10 scores, else stop with an error
            print(f"[sanity] {name}: input {tuple(x.shape)} -> output {tuple(out.shape)} OK")  # tell the user it is fine


# WHAT: Entry point for 'python -m src.models'.
# HOW: Runs the sanity check when the file is executed directly.
# WHY: Quick manual test.
# PROS & CONS: + Standard pattern. - None.
# EXAMPLE: python -m src.models
if __name__ == "__main__":  # true only when this file is run directly
    sanity_check()  # run the quick check
