"""Small helper tools used by the other files."""
import json  # read and write JSON text files
import os  # work with file and folder paths
import random  # Python's random numbers
import shutil  # copy files
import subprocess  # run terminal commands from Python
import sys  # access to the error output

import numpy as np  # fast number lists
import torch  # main library for deep learning

SEED = 42  # one fixed starting number for all randomness, so results can be repeated


def set_seed(seed: int = SEED):  # make all random numbers repeatable
    random.seed(seed)  # fix Python's random numbers
    np.random.seed(seed)  # fix NumPy's random numbers
    torch.manual_seed(seed)  # fix PyTorch's random numbers
    if torch.cuda.is_available():  # if a GPU exists
        torch.cuda.manual_seed_all(seed)  # fix the GPU's random numbers too


def get_device() -> torch.device:  # choose where to run the maths
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU (fast) if there is one, else CPU (slow)


def save_json(obj, path):  # save a Python dictionary or list as a readable text file
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)  # create the folder if it is missing
    with open(path, "w", encoding="utf-8") as f:  # open the file for writing
        json.dump(obj, f, indent=2)  # write the data neatly indented


def load_json(path, default=None):  # read a JSON file; if it does not exist, give back `default`
    if not os.path.exists(path):  # file is missing
        return default  # give the fallback value instead of crashing
    with open(path, "r", encoding="utf-8") as f:  # open the file for reading
        return json.load(f)  # read and return its content


def _exclude_oversized_files(repo_dir, limit_mb=95):  # tell git to ignore saved-model files that are too big for GitHub (limit about 100 MB)
    try:  # never crash because of this
        exclude = os.path.join(repo_dir, ".git", "info", "exclude")  # git's private "ignore" list (not shared)
        if not os.path.isdir(os.path.dirname(exclude)):  # not a git folder
            return  # nothing to do
        existing = open(exclude, encoding="utf-8").read() if os.path.exists(exclude) else ""  # read what is already in the ignore list
        new = []  # big files found
        for root, _dirs, files in os.walk(os.path.join(repo_dir, "checkpoints")):  # look through the saved-models folder
            for f in files:  # check every file
                p = os.path.join(root, f)  # full path of the file
                if f.endswith((".pth", ".pt")) and os.path.getsize(p) > limit_mb * 1024 * 1024:  # a saved-model file bigger than the limit
                    rel = os.path.relpath(p, repo_dir).replace(os.sep, "/")  # its path inside the project
                    if rel not in existing:  # not in the ignore list yet
                        new.append(rel)  # remember it
        if new:  # at least one big file found
            with open(exclude, "a", encoding="utf-8") as fh:  # open the ignore list to add lines
                fh.write("\n" + "\n".join(new) + "\n")  # add the big files
            print(f"[AUTO-PUSH] Excluded from git (>{limit_mb}MB): {', '.join(new)}")  # tell the user
    except OSError:  # a file problem happened
        pass  # ignore it quietly


def git_auto_push(message: str, repo_dir: str = ".") -> bool:  # saves the work to GitHub automatically; never crashes the training
    def run(*cmd):  # small helper that runs one terminal command
        return subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)  # run it and collect its output instead of printing it

    try:  # attempt the upload
        if run("git", "rev-parse", "--is-inside-work-tree").returncode != 0:  # check this folder is a git project
            print("[AUTO-PUSH] Not a git repository, skipping.")  # it is not
            return False  # stop
        _exclude_oversized_files(repo_dir)  # skip files that are too big for GitHub
        run("git", "add", ".")  # mark all changed files to be saved
        if run("git", "diff", "--cached", "--quiet").returncode == 0:  # code 0 means nothing changed
            print("[AUTO-PUSH] No file changes to commit.")  # tell the user
            return True  # nothing to upload
        commit = run("git", "commit", "-m", message)  # save a snapshot with the given message
        if commit.returncode != 0:  # saving the snapshot failed
            print(f"[AUTO-PUSH] Commit failed: {commit.stderr.strip()[:300]}")  # show the reason
            return False  # stop
        push = run("git", "push", "origin", "HEAD")  # upload the snapshot to GitHub
        if push.returncode != 0:  # the upload failed
            print("[AUTO-PUSH] Push failed (check remote / token). Commit kept locally.")  # tell the user (the details are hidden because they may contain secrets)
            return False  # stop
        print(f"[AUTO-PUSH] Successfully pushed to GitHub: {message}")  # tell the user it worked
        return True  # success
    except Exception as e:  # noqa: BLE001 - anything unexpected
        print(f"[AUTO-PUSH] Skipped due to error: {type(e).__name__}")  # tell the user, but keep going
        return False  # stop


def sync_to_drive(src_dir: str, drive_dir: str):  # copy a folder's files to Google Drive, so nothing is lost if Colab disconnects
    if not drive_dir:  # no Drive folder given
        return  # do nothing
    try:  # attempt the copy
        os.makedirs(drive_dir, exist_ok=True)  # create the Drive folder if it is missing
        for name in os.listdir(src_dir):  # go through everything in the source folder
            s = os.path.join(src_dir, name)  # full path of the item
            if os.path.isfile(s):  # only copy files, not folders
                shutil.copy2(s, os.path.join(drive_dir, name))  # copy the file to Drive
    except Exception as e:  # noqa: BLE001 - copying failed
        print(f"[DRIVE] Sync skipped: {type(e).__name__}: {e}", file=sys.stderr)  # tell the user, but keep going


def restore_from_drive(dest_dir: str, drive_dir: str):  # the opposite: bring saved files back from Google Drive
    if not drive_dir or not os.path.isdir(drive_dir):  # no Drive folder to read from
        return  # do nothing
    os.makedirs(dest_dir, exist_ok=True)  # create the local folder if it is missing
    for name in os.listdir(drive_dir):  # go through the files on Drive
        d = os.path.join(dest_dir, name)  # where the file should go here
        s = os.path.join(drive_dir, name)  # where it is on Drive
        if os.path.isfile(s) and not os.path.exists(d):  # it is a file and we do not have it yet
            shutil.copy2(s, d)  # copy it here
            print(f"[DRIVE] Restored {name}")  # tell the user
