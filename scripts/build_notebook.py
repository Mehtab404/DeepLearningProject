"""Builds Adversarial_CNN_vs_ViT_Colab.ipynb from the files in src/ and main.py.

The notebook is self-contained: it re-creates every module with %%writefile, so the one .ipynb is all a
user has to upload to Colab/Kaggle. Re-run this script after editing src/ to keep the notebook in sync:

    python scripts/build_notebook.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODULES = ["src/__init__.py", "src/utils.py", "src/models.py", "src/attacks.py", "src/data.py",
           "src/defense.py", "src/evaluate.py", "src/visualize.py", "main.py"]


def lines(text):
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text.strip("\n"))}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": lines(text.strip("\n"))}


INTRO = """
# Adversarial Attacks & Defences: CNN (ResNet-20) vs. Vision Transformer (ViT) on CIFAR-10

One-click notebook: **Runtime -> Change runtime type -> T4 GPU**, then **Runtime -> Run all**.

Pipeline: pretrained clean CNN/ViT -> native PyTorch FGSM & PGD -> 10-epoch AMP adversarial fine-tuning
(eps = 8/255) -> accuracy-vs-eps curves, cost of adversarial training, transferability, example grid.
Estimated total time on a T4: roughly 20-45 minutes (not yet measured; see README).

**Optional GitHub push:** add a Colab secret named `GITHUB_TOKEN` (fine-grained token, *Contents: read/write*
on the repo only). With it, checkpoints, CSV/MD tables and PNG plots are pushed after every stage and every 5
training epochs. Without it everything still runs, nothing is pushed.
"""

CONFIG = '''
# ---- Configuration -------------------------------------------------------------------------------
import base64, os, subprocess

REPO_URL = "https://github.com/Mehtab404/DeepLearningProject.git"
MOUNT_DRIVE = True          # Colab only: keep checkpoints on Google Drive so a disconnect loses nothing
RUN_QUICK_TEST = False      # True = tiny smoke test first (a few minutes on GPU)
AUTO_PUSH = True            # only has an effect when a GITHUB_TOKEN secret is found
DRIVE_DIR = "/content/drive/MyDrive/adv_cnn_vs_vit_ckpts"

# ---- Optional GITHUB_TOKEN (Colab secret / Kaggle secret / environment variable) -------------------
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
if not GITHUB_TOKEN:
    try:
        from google.colab import userdata
        GITHUB_TOKEN = userdata.get("GITHUB_TOKEN") or ""
    except Exception:
        pass
if not GITHUB_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        GITHUB_TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        pass

# The token is passed to git through environment variables only (never written to .git/config or logs).
if GITHUB_TOKEN:
    basic = base64.b64encode(f"x-access-token:{GITHUB_TOKEN}".encode()).decode()
    os.environ.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.extraHeader",
                       "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}"})
    print("GITHUB_TOKEN found: results will be pushed to GitHub.")
else:
    AUTO_PUSH = False
    print("No GITHUB_TOKEN found: running without GitHub pushes.")

# ---- Working directory = a clone of the repo (so pushes land in the right place) ------------------
BASE = "/content" if os.path.isdir("/content") else ("/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd())
REPO_DIR = os.path.join(BASE, "DeepLearningProject")
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    r = subprocess.run(["git", "clone", REPO_URL, REPO_DIR], capture_output=True, text=True)
    if r.returncode != 0:
        print("Clone failed (private repo without token?). Continuing in a fresh local folder, pushes disabled.")
        os.makedirs(REPO_DIR, exist_ok=True)
        subprocess.run(["git", "init", "-b", "main"], cwd=REPO_DIR, capture_output=True)
        AUTO_PUSH = False
os.chdir(REPO_DIR)
subprocess.run(["git", "config", "user.name", "colab-agent"])
subprocess.run(["git", "config", "user.email", "colab-agent@users.noreply.github.com"])
os.makedirs("src", exist_ok=True); os.makedirs("checkpoints", exist_ok=True); os.makedirs("outputs", exist_ok=True)
print("Working in", os.getcwd())

if MOUNT_DRIVE and os.path.isdir("/content"):
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive mount skipped:", type(e).__name__)
        DRIVE_DIR = ""
else:
    DRIVE_DIR = ""
'''

INSTALL = """
!pip -q install timm transformers seaborn tqdm pandas
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only (use --quick-test)")
"""

SANITY = """
# Task 2.4 / 3.4 sanity checks: model output shapes and attack budget/range assertions
from src.models import sanity_check
from src.attacks import self_test
sanity_check()
self_test()
"""

QUICK = """
if RUN_QUICK_TEST:
    !python main.py --quick-test
"""

RUN = """
# Full run: loads checkpoints/*.pth if they exist (Drive is restored first), otherwise trains.
extra = ""
if DRIVE_DIR: extra += f" --drive-dir '{DRIVE_DIR}'"
if AUTO_PUSH: extra += " --auto-push"
!python main.py --full-run --epochs 10 --eval-samples 1000 {extra}
"""

SHOW = """
import json
from IPython.display import Image, Markdown, display
info = json.load(open("outputs/run_info.json"))
assert info["mode"] == "full-run", "outputs/ holds a quick-test run, not full-run results. Re-run the training cell."
for f in ["acc_vs_epsilon", "clean_vs_robust_tradeoff", "transferability_heatmap", "adversarial_examples_grid"]:
    display(Image(filename=f"outputs/{f}.png"))
display(Markdown(open("outputs/results_summary.md", encoding="utf-8").read()))
"""

FINAL_PUSH = """
from src.utils import git_auto_push
if AUTO_PUSH:
    git_auto_push("results(final): push complete CIFAR-10 CNN vs ViT robustness metrics and 300-DPI plots [Milestone 7]")
"""


def build():
    cells = [md(INTRO), code(CONFIG), code(INSTALL)]
    for rel in MODULES:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            src = f.read().rstrip("\n")
        cells.append(code(f"%%writefile {rel}\n{src}"))
    cells += [md("## Sanity checks"), code(SANITY), md("## Optional quick smoke test"), code(QUICK),
              md("## Full run (train + evaluate + plots)"), code(RUN),
              md("## Results"), code(SHOW), code(FINAL_PUSH)]
    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    # nbformat >= 4.5 requires a unique id per cell
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell{i:03d}"
    out = os.path.join(ROOT, "Adversarial_CNN_vs_ViT_Colab.ipynb")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
