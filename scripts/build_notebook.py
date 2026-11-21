"""Builds Adversarial_CNN_vs_ViT_Colab.ipynb from the files in src/ and main.py.

The notebook is self-contained: it re-creates every module with %%writefile, so the one .ipynb is all a
user has to upload to Colab/Kaggle. Re-run this script after editing src/ to keep the notebook in sync:

    python scripts/build_notebook.py
"""
import json  # write the notebook file (a notebook is a JSON text file)
import os  # work with file and folder paths

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # the main project folder (one level above scripts/)
MODULES = ["src/__init__.py", "src/utils.py", "src/models.py", "src/attacks.py", "src/data.py",  # the code files that are copied into the notebook, in this order
           "src/defense.py", "src/evaluate.py", "src/visualize.py", "main.py"]  # the rest of the files


def lines(text):  # splits text into lines, the way a notebook stores them
    parts = text.split("\n")  # cut the text at every line break
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])  # put the line break back on every line except the last


def md(text):  # makes a notebook cell that shows plain text (not code)
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text.strip("\n"))}  # a "markdown" cell holding the text


def code(text):  # makes a notebook cell that holds code
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],  # a "code" cell that has not been run yet, so no output
            "source": lines(text.strip("\n"))}  # the code, split into lines


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
import base64, os, subprocess  # base64 = text encoding; os = files and folders; subprocess = run terminal commands

REPO_URL = "https://github.com/Mehtab404/DeepLearningProject.git"  # the GitHub project this notebook uses
MOUNT_DRIVE = True  # Colab only: keep saved models on Google Drive so a disconnect loses nothing
RUN_QUICK_TEST = False  # True = run a tiny test first (a few minutes on a GPU)
AUTO_PUSH = True  # upload results to GitHub; only works when a GITHUB_TOKEN secret is found
DRIVE_DIR = "/content/drive/MyDrive/adv_cnn_vs_vit_ckpts"  # Google Drive folder for the backups

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")  # first try: read the token (GitHub's secret password) from the computer's settings
if not GITHUB_TOKEN:  # not found there
    try:  # second try: Colab's secret storage
        from google.colab import userdata  # Colab's secret storage
        GITHUB_TOKEN = userdata.get("GITHUB_TOKEN") or ""  # read the secret (empty text if missing)
    except Exception:  # we are not on Colab
        pass  # ignore it
if not GITHUB_TOKEN:  # still not found
    try:  # third try: Kaggle's secret storage
        from kaggle_secrets import UserSecretsClient  # Kaggle's secret storage
        GITHUB_TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")  # read the secret
    except Exception:  # we are not on Kaggle
        pass  # ignore it

if GITHUB_TOKEN:  # a token was found
    basic = base64.b64encode(f"x-access-token:{GITHUB_TOKEN}".encode()).decode()  # turn the token into the text format git's login needs
    os.environ.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.extraHeader",  # give the login to git through temporary settings (never saved in a file or log)
                       "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}"})  # the login text itself
    print("GITHUB_TOKEN found: results will be pushed to GitHub.")  # tell the user
else:  # no token
    AUTO_PUSH = False  # switch uploading off
    print("No GITHUB_TOKEN found: running without GitHub pushes.")  # tell the user

BASE = "/content" if os.path.isdir("/content") else ("/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd())  # the working folder: Colab's, else Kaggle's, else the current one
REPO_DIR = os.path.join(BASE, "DeepLearningProject")  # folder where the project will be placed
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):  # the project is not downloaded yet
    r = subprocess.run(["git", "clone", REPO_URL, REPO_DIR], capture_output=True, text=True)  # download (clone) the project from GitHub
    if r.returncode != 0:  # the download failed
        print("Clone failed (private repo without token?). Continuing in a fresh local folder, pushes disabled.")  # tell the user
        os.makedirs(REPO_DIR, exist_ok=True)  # create an empty folder instead
        subprocess.run(["git", "init", "-b", "main"], cwd=REPO_DIR, capture_output=True)  # start a new empty git project there
        AUTO_PUSH = False  # switch uploading off
os.chdir(REPO_DIR)  # move into the project folder
subprocess.run(["git", "config", "user.name", "colab-agent"])  # name shown on the uploads
subprocess.run(["git", "config", "user.email", "colab-agent@users.noreply.github.com"])  # email shown on the uploads
os.makedirs("src", exist_ok=True); os.makedirs("checkpoints", exist_ok=True); os.makedirs("outputs", exist_ok=True)  # create the folders for code, saved models and results
print("Working in", os.getcwd())  # show the working folder

if MOUNT_DRIVE and os.path.isdir("/content"):  # on Colab and Drive is wanted
    try:  # attempt to connect Google Drive
        from google.colab import drive  # Colab's Drive tool
        drive.mount("/content/drive")  # connect Google Drive (asks for permission)
    except Exception as e:  # it did not work
        print("Drive mount skipped:", type(e).__name__)  # tell the user
        DRIVE_DIR = ""  # do not use Drive
else:  # not on Colab, or Drive not wanted
    DRIVE_DIR = ""  # do not use Drive
'''

INSTALL = """
# next line installs the extra libraries the project needs
!pip -q install timm transformers seaborn tqdm pandas
import torch  # main library for deep learning
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),  # show the library version and whether a GPU is available
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only (use --quick-test)")  # show the GPU name, or a warning that only the CPU is available
"""

SANITY = """
from src.models import sanity_check  # the quick model check
from src.attacks import self_test  # the quick attack check
sanity_check()  # check both models give 10 scores per image
self_test()  # check the attacks follow their rules
"""

QUICK = """
if RUN_QUICK_TEST:  # only if the tiny test was asked for
    # next line runs the tiny test
    !python main.py --quick-test
"""

RUN = """
extra = ""  # extra options to add to the command below
if DRIVE_DIR: extra += f" --drive-dir '{DRIVE_DIR}'"  # add the Google Drive backup folder if we have one
if AUTO_PUSH: extra += " --auto-push"  # add automatic GitHub upload if switched on
# next line runs the whole project: loads saved models if they exist, otherwise trains, then tests and draws graphs
!python main.py --full-run --epochs 10 --eval-samples 1000 {extra}
"""

SHOW = """
import json  # read JSON text files
from IPython.display import Image, Markdown, display  # tools to show pictures and text in the notebook
info = json.load(open("outputs/run_info.json"))  # read the summary of the last run
assert info["mode"] == "full-run", "outputs/ holds a quick-test run, not full-run results. Re-run the training cell."  # stop with an error if the results are only from the tiny test
for f in ["acc_vs_epsilon", "clean_vs_robust_tradeoff", "transferability_heatmap", "adversarial_examples_grid"]:  # the four graphs
    display(Image(filename=f"outputs/{f}.png"))  # show this graph
display(Markdown(open("outputs/results_summary.md", encoding="utf-8").read()))  # show the results tables
"""

FINAL_PUSH = """
from src.utils import git_auto_push  # the function that uploads to GitHub
if AUTO_PUSH:  # only if uploading is switched on
    git_auto_push("results(final): push complete CIFAR-10 CNN vs ViT robustness metrics and 300-DPI plots [Milestone 7]")  # upload the final results
"""


def build():  # creates the notebook file
    cells = [md(INTRO), code(CONFIG), code(INSTALL)]  # first cells: introduction text, setup code, install code
    for rel in MODULES:  # go through each code file
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:  # open the file
            src = f.read().rstrip("\n")  # read all its text (without empty lines at the end)
        cells.append(code(f"%%writefile {rel}\n{src}"))  # add a cell that, when run, recreates that file inside Colab
    cells += [md("## Sanity checks"), code(SANITY), md("## Optional quick smoke test"), code(QUICK),  # next cells: quick checks, then the optional tiny test
              md("## Full run (train + evaluate + plots)"), code(RUN),  # the main run
              md("## Results"), code(SHOW), code(FINAL_PUSH)]  # show the results, then upload them
    nb = {"cells": cells,  # the whole notebook: all the cells
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},  # ask for a GPU (T4) on Colab
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},  # run the code with Python 3
                       "language_info": {"name": "python"}},  # the language is Python
          "nbformat": 4, "nbformat_minor": 5}  # notebook file format version
    for i, c in enumerate(nb["cells"]):  # go through every cell
        c["id"] = f"cell{i:03d}"  # give it a unique id (newer notebooks need one)
    out = os.path.join(ROOT, "Adversarial_CNN_vs_ViT_Colab.ipynb")  # where to save the notebook
    with open(out, "w", encoding="utf-8", newline="\n") as f:  # open the file for writing
        json.dump(nb, f, indent=1, ensure_ascii=False)  # write the notebook as JSON text
        f.write("\n")  # end with an empty line
    print(f"Wrote {out} ({len(cells)} cells)")  # tell the user it worked


if __name__ == "__main__":  # true only when this file is run directly
    build()  # create the notebook
