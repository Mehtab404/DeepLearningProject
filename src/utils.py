"""Small helper tools used by the other files."""
# WHAT: Imports libraries for JSON files, paths, random numbers, file copying, terminal commands and arrays.
# HOW: json = text format for logs, subprocess = run git, shutil = copy files, numpy/random/torch = seeds.
# WHY: These small helpers are shared by training, evaluation and the notebook.
# PROS & CONS: + Standard library. - None.
# EXAMPLE: json.dump(obj, f) writes the training history.
import json  # read and write JSON text files
import os  # work with file and folder paths
import random  # Python's random numbers
import shutil  # copy files
import subprocess  # run terminal commands from Python
import sys  # access to the error output

import numpy as np  # fast number lists
import torch  # main library for deep learning

# WHAT: Single project-wide random seed.
# HOW: A constant number used to initialise every random generator.
# WHY: Same seed gives the same evaluation subset and the same random starts (when the call order is the same).
# PROS & CONS: + Repeatable. - A single seed does not measure variation between training runs.
# EXAMPLE: SEED = 42.
SEED = 42  # one fixed starting number for all randomness, so results can be repeated


# WHAT: Fixes all random generators.
# HOW: Seeds Python's random, numpy, PyTorch CPU and all GPUs.
# WHY: Different libraries have separate generators; all must be fixed.
# PROS & CONS: + Repeatable runs. - GPU non-determinism can still cause tiny differences.
# EXAMPLE: set_seed(42).
def set_seed(seed: int = SEED):  # make all random numbers repeatable
    random.seed(seed)  # fix Python's random numbers
    np.random.seed(seed)  # fix NumPy's random numbers
    torch.manual_seed(seed)  # fix PyTorch's random numbers
    if torch.cuda.is_available():  # if a GPU exists
        torch.cuda.manual_seed_all(seed)  # fix the GPU's random numbers too


# WHAT: Chooses where to compute.
# HOW: Returns cuda if a GPU is visible, otherwise cpu.
# WHY: Lets the same code run on Colab (GPU) and a laptop (CPU).
# PROS & CONS: + Portable. - CPU runs are very slow for full training.
# EXAMPLE: device(type='cuda') on a Colab T4.
def get_device() -> torch.device:  # choose where to run the maths
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU (fast) if there is one, else CPU (slow)


# WHAT: Writes a Python dictionary or list to a readable text file.
# HOW: Creates the folder if needed, then json.dump with indent=2.
# WHY: Training history and run summaries must survive Colab disconnects.
# PROS & CONS: + Human-readable. - Not atomic: a crash mid-write could corrupt the file.
# EXAMPLE: save_json({'epoch': 1}, 'checkpoints/training_history.json').
def save_json(obj, path):  # save a Python dictionary or list as a readable text file
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)  # create the folder if it is missing
    with open(path, "w", encoding="utf-8") as f:  # open the file for writing
        json.dump(obj, f, indent=2)  # write the data neatly indented


# WHAT: Reads a JSON file, or returns a default if it does not exist.
# HOW: os.path.exists check, then json.load.
# WHY: The first run has no history file yet and must not crash.
# PROS & CONS: + Safe default. - Silently hides a missing file.
# EXAMPLE: load_json('missing.json', {}) returns {}.
def load_json(path, default=None):  # read a JSON file; if it does not exist, give back `default`
    if not os.path.exists(path):  # file is missing
        return default  # give the fallback value instead of crashing
    with open(path, "r", encoding="utf-8") as f:  # open the file for reading
        return json.load(f)  # read and return its content


# WHAT: Keeps weight files that are too big for GitHub out of git.
# HOW: Scans checkpoints/ for .pth/.pt files above 95 MB and appends them to .git/info/exclude (a private ignore
#      list).
# WHY: GitHub rejects files above 100 MB; one huge file would make every push fail.
# PROS & CONS: + Automatic. - Silent exclusion means large weights must be backed up elsewhere (Drive).
# EXAMPLE: A ViT-Base weight file of about 340 MB is excluded; a 22 MB ViT-Tiny weight file is not.
def _exclude_oversized_files(repo_dir, limit_mb=95):  # tell git to ignore saved-model files that are too big for GitHub (limit about 100 MB)
    # WHAT: Wraps the whole job so it can never crash training.
    # HOW: Any OSError is ignored at the end.
    # WHY: Saving results must never kill a long training run.
    # PROS & CONS: + Robust. - Hides failures.
    # EXAMPLE: A read-only folder just skips the step.
    try:  # never crash because of this
        exclude = os.path.join(repo_dir, ".git", "info", "exclude")  # git's private "ignore" list (not shared)
        if not os.path.isdir(os.path.dirname(exclude)):  # not a git folder
            return  # nothing to do
        existing = open(exclude, encoding="utf-8").read() if os.path.exists(exclude) else ""  # read what is already in the ignore list
        new = []  # big files found
        # WHAT: Looks through every file in checkpoints/.
        # HOW: os.walk lists each folder; the size is compared with the limit in bytes.
        # WHY: Finds the files that are too large.
        # PROS & CONS: + Complete scan. - Walks the folder every push (cheap).
        # EXAMPLE: 95 * 1024 * 1024 bytes = 99,614,720.
        for root, _dirs, files in os.walk(os.path.join(repo_dir, "checkpoints")):  # look through the saved-models folder
            for f in files:  # check every file
                p = os.path.join(root, f)  # full path of the file
                if f.endswith((".pth", ".pt")) and os.path.getsize(p) > limit_mb * 1024 * 1024:  # a saved-model file bigger than the limit
                    rel = os.path.relpath(p, repo_dir).replace(os.sep, "/")  # its path inside the project
                    if rel not in existing:  # not in the ignore list yet
                        new.append(rel)  # remember it
        # WHAT: Writes the new exclusions.
        # HOW: Appends the paths to .git/info/exclude and prints them.
        # WHY: Tells the user which files will not be pushed.
        # PROS & CONS: + Visible. - None.
        # EXAMPLE: [AUTO-PUSH] Excluded from git (>95MB): checkpoints/big.pth
        if new:  # at least one big file found
            with open(exclude, "a", encoding="utf-8") as fh:  # open the ignore list to add lines
                fh.write("\n" + "\n".join(new) + "\n")  # add the big files
            print(f"[AUTO-PUSH] Excluded from git (>{limit_mb}MB): {', '.join(new)}")  # tell the user
    except OSError:  # a file problem happened
        pass  # ignore it quietly


# WHAT: Commits and pushes all changes to GitHub, never raising an error.
# HOW: Runs git add, commit and push through subprocess; every failure is printed and returns False.
# WHY: Colab can disconnect at any time; pushing after each stage protects the results.
# PROS & CONS: + Automatic backup. - git add . stages everything in the folder (including unwanted files if
#              .gitignore is incomplete).
# EXAMPLE: git_auto_push('results: evaluation tables and plots').
def git_auto_push(message: str, repo_dir: str = ".") -> bool:  # saves the work to GitHub automatically; never crashes the training
    # WHAT: Small helper that runs one terminal command.
    # HOW: subprocess.run with captured output so nothing is printed (tokens could appear in output).
    # WHY: Keeps secrets out of the logs.
    # PROS & CONS: + Safe logs. - Errors are only visible through return codes.
    # EXAMPLE: run('git', 'status').
    def run(*cmd):  # small helper that runs one terminal command
        return subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)  # run it and collect its output instead of printing it

    # WHAT: Starts the guarded upload.
    # HOW: Everything below is inside a try so unexpected errors are caught.
    # WHY: A failed push should not stop the experiment.
    # PROS & CONS: + Robust. - May hide bugs.
    # EXAMPLE: A network error prints '[AUTO-PUSH] Push failed'.
    try:  # attempt the upload
        # WHAT: Checks the folder is a git repository.
        # HOW: git rev-parse --is-inside-work-tree returns 0 inside a repo.
        # WHY: Avoids confusing errors outside git.
        # PROS & CONS: + Clear message. - None.
        # EXAMPLE: '[AUTO-PUSH] Not a git repository, skipping.'
        if run("git", "rev-parse", "--is-inside-work-tree").returncode != 0:  # check this folder is a git project
            print("[AUTO-PUSH] Not a git repository, skipping.")  # it is not
            return False  # stop
        # WHAT: Excludes big files before staging.
        # HOW: Calls the helper above.
        # WHY: Prevents a rejected push.
        # PROS & CONS: + Safe. - None.
        # EXAMPLE: Run before git add.
        _exclude_oversized_files(repo_dir)  # skip files that are too big for GitHub
        # WHAT: Stages all changes.
        # HOW: git add . adds every new and modified file.
        # WHY: Simple for a research repository.
        # PROS & CONS: + Convenient. - Can add junk files; .gitignore must be correct.
        # EXAMPLE: A new PNG plot is staged.
        run("git", "add", ".")  # mark all changed files to be saved
        # WHAT: Skips empty commits.
        # HOW: git diff --cached --quiet returns 0 when nothing is staged.
        # WHY: No point committing nothing.
        # PROS & CONS: + Clean history. - None.
        # EXAMPLE: '[AUTO-PUSH] No file changes to commit.'
        if run("git", "diff", "--cached", "--quiet").returncode == 0:  # code 0 means nothing changed
            print("[AUTO-PUSH] No file changes to commit.")  # tell the user
            return True  # nothing to upload
        # WHAT: Creates the commit.
        # HOW: git commit -m message.
        # WHY: Records a snapshot with a descriptive message.
        # PROS & CONS: + Traceable history. - None.
        # EXAMPLE: 'chore(train): CNN-AdvTrained 10-epoch adversarial fine-tuning finished'.
        commit = run("git", "commit", "-m", message)  # save a snapshot with the given message
        if commit.returncode != 0:  # saving the snapshot failed
            print(f"[AUTO-PUSH] Commit failed: {commit.stderr.strip()[:300]}")  # show the reason
            return False  # stop
        # WHAT: Uploads the commit to GitHub.
        # HOW: git push origin HEAD; failure details are hidden because they may contain a token.
        # WHY: Protects the credentials.
        # PROS & CONS: + No secrets in logs. - Less debugging detail.
        # EXAMPLE: '[AUTO-PUSH] Successfully pushed to GitHub: ...'
        push = run("git", "push", "origin", "HEAD")  # upload the snapshot to GitHub
        if push.returncode != 0:  # the upload failed
            print("[AUTO-PUSH] Push failed (check remote / token). Commit kept locally.")  # tell the user (the details are hidden because they may contain secrets)
            return False  # stop
        print(f"[AUTO-PUSH] Successfully pushed to GitHub: {message}")  # tell the user it worked
        return True  # success
    # WHAT: Catch-all for unexpected errors.
    # HOW: Prints only the exception type, then returns False.
    # WHY: Never crash the experiment because of git.
    # PROS & CONS: + Robust. - May hide real bugs.
    # EXAMPLE: '[AUTO-PUSH] Skipped due to error: FileNotFoundError'.
    except Exception as e:  # noqa: BLE001 - anything unexpected
        print(f"[AUTO-PUSH] Skipped due to error: {type(e).__name__}")  # tell the user, but keep going
        return False  # stop


# WHAT: Copies a folder's files to Google Drive.
# HOW: Creates the Drive folder, copies each file with copy2.
# WHY: Colab local disk is wiped when the session ends; Drive survives.
# PROS & CONS: + Resume after disconnect. - Overwrites Drive files each call.
# EXAMPLE: sync_to_drive('checkpoints', '/content/drive/MyDrive/adv_ckpts').
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


# WHAT: The reverse: brings missing files back from Drive.
# HOW: Copies only files that do not exist locally.
# WHY: Lets a new Colab session continue where the old one stopped.
# PROS & CONS: + No retraining. - Does not update older local copies.
# EXAMPLE: '[DRIVE] Restored cnn_adv.pth'.
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
