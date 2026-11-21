"""Small shared helpers: seeding, device selection, JSON I/O, git auto-push, Drive sync."""
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import torch

# ===================================================================================================
# EASY-LANGUAGE SUMMARY OF THIS FILE (read this first)
# ---------------------------------------------------------------------------------------------------
# What is this file?   Small helper tools used by the other files (no deep learning here).
#   set_seed()          - makes random numbers repeatable, so every run gives the same result.
#   get_device()        - chooses GPU (fast) if available, otherwise CPU (slow).
#   save_json/load_json - save / read results in a simple text file format (JSON).
#   git_auto_push()     - automatically uploads the work to GitHub (optional, for Colab).
#   sync_to_drive / restore_from_drive - back up files to Google Drive and bring them back.
# ===================================================================================================

# One fixed random seed for the whole project, so results can be reproduced.
SEED = 42


def set_seed(seed: int = SEED):
    # Python, NumPy and PyTorch each have their own random number generator. Seeding all of them
    # makes shuffling, random crops and attack random starts the same on every run.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    # Use the GPU ("cuda") if there is one, otherwise the CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    # Write a Python dict/list to a readable JSON file (creates the folder if needed).
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path, default=None):
    # Read a JSON file. If it does not exist yet, return `default` instead of crashing.
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _exclude_oversized_files(repo_dir, limit_mb=95):
    """GitHub rejects files > 100MB. Add any .pth/.pt above `limit_mb` to .git/info/exclude."""
    # .git/info/exclude works like .gitignore but stays local (it is never committed).
    try:
        exclude = os.path.join(repo_dir, ".git", "info", "exclude")
        if not os.path.isdir(os.path.dirname(exclude)):
            return
        existing = open(exclude, encoding="utf-8").read() if os.path.exists(exclude) else ""
        new = []
        # Walk through every file in checkpoints/ and collect the ones that are too big.
        for root, _dirs, files in os.walk(os.path.join(repo_dir, "checkpoints")):
            for f in files:
                p = os.path.join(root, f)
                if f.endswith((".pth", ".pt")) and os.path.getsize(p) > limit_mb * 1024 * 1024:
                    rel = os.path.relpath(p, repo_dir).replace(os.sep, "/")
                    if rel not in existing:
                        new.append(rel)
        if new:
            with open(exclude, "a", encoding="utf-8") as fh:
                fh.write("\n" + "\n".join(new) + "\n")
            print(f"[AUTO-PUSH] Excluded from git (>{limit_mb}MB): {', '.join(new)}")
    except OSError:
        pass


def git_auto_push(message: str, repo_dir: str = ".") -> bool:
    """Python equivalent of scripts/auto_push.sh. Never raises: training must not die on a git error.

    Stages everything, commits if there is something to commit, and pushes HEAD to origin.
    """
    # Used with --auto-push: saves progress to GitHub during long Colab runs.

    def run(*cmd):
        # Run a terminal command (e.g. git status) and capture its output instead of printing it.
        return subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)

    try:
        if run("git", "rev-parse", "--is-inside-work-tree").returncode != 0:
            print("[AUTO-PUSH] Not a git repository, skipping.")
            return False
        _exclude_oversized_files(repo_dir)
        run("git", "add", ".")
        # "git diff --cached --quiet" returns 0 when nothing is staged -> nothing to commit.
        if run("git", "diff", "--cached", "--quiet").returncode == 0:
            print("[AUTO-PUSH] No file changes to commit.")
            return True
        commit = run("git", "commit", "-m", message)
        if commit.returncode != 0:
            print(f"[AUTO-PUSH] Commit failed: {commit.stderr.strip()[:300]}")
            return False
        push = run("git", "push", "origin", "HEAD")
        if push.returncode != 0:
            # stderr can echo the remote URL; never print it in full.
            print("[AUTO-PUSH] Push failed (check remote / token). Commit kept locally.")
            return False
        print(f"[AUTO-PUSH] Successfully pushed to GitHub: {message}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[AUTO-PUSH] Skipped due to error: {type(e).__name__}")
        return False


def sync_to_drive(src_dir: str, drive_dir: str):
    """Copy a directory tree (e.g. checkpoints/) to Google Drive so a disconnect never loses work."""
    # Colab deletes local files when the session ends; Google Drive keeps them.
    if not drive_dir:
        return    # not on Colab / no Drive folder given -> do nothing
    try:
        os.makedirs(drive_dir, exist_ok=True)
        for name in os.listdir(src_dir):
            s = os.path.join(src_dir, name)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(drive_dir, name))
    except Exception as e:  # noqa: BLE001
        print(f"[DRIVE] Sync skipped: {type(e).__name__}: {e}", file=sys.stderr)


def restore_from_drive(dest_dir: str, drive_dir: str):
    """Copy files from Drive into dest_dir if they are not already there."""
    # The opposite of sync_to_drive: at the start of a new Colab session, bring back saved checkpoints.
    if not drive_dir or not os.path.isdir(drive_dir):
        return
    os.makedirs(dest_dir, exist_ok=True)
    for name in os.listdir(drive_dir):
        d = os.path.join(dest_dir, name)
        s = os.path.join(drive_dir, name)
        if os.path.isfile(s) and not os.path.exists(d):
            shutil.copy2(s, d)
            print(f"[DRIVE] Restored {name}")
