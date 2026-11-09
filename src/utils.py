"""Small shared helpers: seeding, device selection, JSON I/O, git auto-push, Drive sync."""
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_auto_push(message: str, repo_dir: str = ".") -> bool:
    """Python equivalent of scripts/auto_push.sh. Never raises: training must not die on a git error.

    Stages everything, commits if there is something to commit, and pushes HEAD to origin.
    """

    def run(*cmd):
        return subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)

    try:
        if run("git", "rev-parse", "--is-inside-work-tree").returncode != 0:
            print("[AUTO-PUSH] Not a git repository, skipping.")
            return False
        run("git", "add", ".")
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
    if not drive_dir:
        return
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
    if not drive_dir or not os.path.isdir(drive_dir):
        return
    os.makedirs(dest_dir, exist_ok=True)
    for name in os.listdir(drive_dir):
        d = os.path.join(dest_dir, name)
        s = os.path.join(drive_dir, name)
        if os.path.isfile(s) and not os.path.exists(d):
            shutil.copy2(s, d)
            print(f"[DRIVE] Restored {name}")
