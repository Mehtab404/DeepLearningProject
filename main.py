"""CLI entry point: CNN vs ViT adversarial robustness on CIFAR-10.

  python main.py --quick-test            # CPU smoke test (64 test images, 1 tiny epoch), no GPU needed
  python main.py --full-run              # Colab/Kaggle GPU: 10-epoch adversarial fine-tuning + full evaluation
  python main.py --epochs 10 --eval-samples 1000 --auto-push

Checkpoints in checkpoints/ are reused automatically: nothing that already exists is retrained.
"""
# WHAT: Imports the standard libraries used by the command-line program.
# HOW: argparse reads options, copy makes independent model copies, json prints results, os handles paths, time
#      measures the run.
# WHY: main.py coordinates training and evaluation and needs these basic tools.
# PROS & CONS: + Standard library only. - None.
# EXAMPLE: copy.deepcopy(model) clones a model.
import argparse  # read options typed in the terminal
import copy  # make independent copies of models
import json  # print / save results as JSON text
import os  # work with file and folder paths
import time  # measure how long the run takes

# WHAT: Imports the project's own building blocks.
# HOW: Data loading, training, evaluation, model loading, utilities and plotting come from src/.
# WHY: main.py contains no algorithms; it only orders the steps.
# PROS & CONS: + Clear separation. - Import errors appear at start-up if a file is missing.
# EXAMPLE: from src.attacks is imported indirectly by defense.py and evaluate.py.
from src.data import get_eval_tensors, get_train_loader, prepare_data  # functions that get the pictures
from src.defense import adversarial_finetune, clean_finetune, is_training_complete  # functions that train the models
from src.evaluate import accuracy, evaluate_epsilon_sweep, evaluate_transferability, export_results  # functions that test the models and save results
from src.models import load_checkpoint, load_cnn_clean, load_vit_clean, save_checkpoint  # functions that create, load and save models
from src.utils import (SEED, get_device, git_auto_push, load_json, restore_from_drive, save_json, set_seed,  # small helper tools
                       sync_to_drive)  # backup to Google Drive
from src.visualize import make_all_plots  # function that draws the graphs


# WHAT: Defines the command-line options.
# HOW: argparse turns text such as '--epochs 5' into attributes of an args object.
# WHY: The same file serves a 2-minute smoke test and the full GPU run.
# PROS & CONS: + Flexible, self-documenting (--help). - Many options to remember.
# EXAMPLE: python main.py --full-run --epochs 10 --eval-samples 1000
def parse_args():  # reads the options typed after "python main.py", e.g. --epochs 5
    # WHAT: Creates the option reader.
    # HOW: RawDescriptionHelpFormatter prints the module docstring unchanged for --help.
    # WHY: Keeps the usage examples readable.
    # PROS & CONS: + Good --help text. - None.
    # EXAMPLE: python main.py --help
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # the option reader; shows the text at the top of this file when asked for help
    # WHAT: Groups the two run modes so only one can be chosen.
    # HOW: argparse rejects '--quick-test --full-run'.
    # WHY: A run is either a smoke test or a real run.
    # PROS & CONS: + Prevents contradictory options. - None.
    # EXAMPLE: python main.py --quick-test
    mode = p.add_mutually_exclusive_group()  # options in this group cannot be used together
    # WHAT: Option for the tiny CPU smoke test.
    # HOW: Flag without value; True when given.
    # WHY: Checks the whole pipeline in minutes before a long GPU run.
    # PROS & CONS: + Cheap safety net. - Quick-test numbers are NOT results (marked on the plots).
    # EXAMPLE: 64 test images, 1 epoch, 2 PGD steps.
    mode.add_argument("--quick-test", action="store_true", help="CPU smoke test: 64 images, 1 short epoch")  # tiny fast run just to check the code works
    # WHAT: Option for the real run (the default).
    # HOW: Flag without value.
    # WHY: Makes the intention explicit in notebooks and logs.
    # PROS & CONS: + Clear. - None.
    # EXAMPLE: python main.py --full-run
    mode.add_argument("--full-run", action="store_true", help="Full GPU run (default when no mode is given)")  # the real run
    # WHAT: Number of adversarial fine-tuning epochs.
    # HOW: Integer, default 10.
    # WHY: 10 epochs from pretrained weights are enough to see the robustness gain and fit a free GPU.
    # PROS & CONS: + Cheap. - Shorter than training robust models from scratch.
    # EXAMPLE: --epochs 10
    p.add_argument("--epochs", type=int, default=10, help="adversarial fine-tuning epochs")  # how many times the model sees the whole training set
    # WHAT: Size of the stratified test subset.
    # HOW: Integer, default 1000 (100 per class).
    # WHY: Attacks are expensive; 1000 images give a standard error of up to 1.6 points.
    # PROS & CONS: + Fast evaluation. - Not the full 10,000 test images.
    # EXAMPLE: --eval-samples 1000
    p.add_argument("--eval-samples", type=int, default=1000, help="stratified test subset size")  # how many test images are used for testing
    # WHAT: Training epsilon in /255 units.
    # HOW: Float, default 8.
    # WHY: 8/255 is the common CIFAR-10 benchmark budget.
    # PROS & CONS: + Standard. - Robustness at other budgets is weaker (e.g. 16/255).
    # EXAMPLE: --eps 8 means 8/255 = 0.0314.
    p.add_argument("--eps", type=float, default=8.0, help="training epsilon in /255 units")  # biggest change per pixel used while training the defence
    # WHAT: Which attack trains the CNN defence.
    # HOW: Choice of pgd5 or rs_fgsm.
    # WHY: PGD-5 is the stronger, safer choice.
    # PROS & CONS: + Switchable. - rs_fgsm risks catastrophic overfitting.
    # EXAMPLE: default pgd5.
    p.add_argument("--cnn-method", choices=["pgd5", "rs_fgsm"], default="pgd5")  # which attack trains the CNN's defence
    # WHAT: Which attack trains the ViT defence.
    # HOW: Same choices; default pgd5 so the CNN and the ViT are treated alike.
    # WHY: The first ViT run with rs_fgsm showed catastrophic overfitting (79.8% FGSM, 4.9% PGD-10).
    # PROS & CONS: + Fair comparison. - PGD-5 is about 2.5x slower for the ViT (74.7 vs 30.1 minutes).
    # EXAMPLE: default pgd5.
    p.add_argument("--vit-method", choices=["pgd5", "rs_fgsm"], default="pgd5",  # which attack trains the ViT's defence
                   help="pgd5 matches the CNN (fair comparison). rs_fgsm is ~2.5x faster but the ViT showed "  # help text, part 1
                        "signs of catastrophic overfitting with it (see outputs_vit_rsfgsm/)")  # help text, part 2
    # WHAT: Which ViT to use.
    # HOW: timm-tiny (5.7M weights) or hf-base (much larger).
    # WHY: Tiny is the only option that fits a free Colab session.
    # PROS & CONS: + Choice. - hf-base is slow and its weights exceed GitHub's file limit.
    # EXAMPLE: default timm-tiny.
    p.add_argument("--vit-backbone", choices=["timm-tiny", "hf-base"], default="timm-tiny")  # which ViT to use: small (fast) or big (slow)
    # WHAT: Epochs of normal fine-tuning for the new ViT head.
    # HOW: Integer, default 5.
    # WHY: A new 10-class head must learn CIFAR-10 before adversarial training.
    # PROS & CONS: + Reaches 97.1% clean accuracy. - Extra time.
    # EXAMPLE: --vit-clean-epochs 5
    p.add_argument("--vit-clean-epochs", type=int, default=5, help="clean fine-tune epochs for timm ViT-Tiny")  # epochs of normal training for the small ViT
    # WHAT: Learning rate.
    # HOW: Float, default 1e-4.
    # WHY: Gentle updates for fine-tuning pretrained weights.
    # PROS & CONS: + Stable. - May be too small for training from scratch.
    # EXAMPLE: --lr 0.0001
    p.add_argument("--lr", type=float, default=1e-4)  # learning rate = how big each weight update is
    # WHAT: Images per training step.
    # HOW: Integer, default 128.
    # WHY: Fits the T4 even with the doubled (clean + adversarial) batch.
    # PROS & CONS: + Good speed/memory balance. - Larger batches need a larger learning rate.
    # EXAMPLE: 128 images, 256 including adversarial copies.
    p.add_argument("--batch-size", type=int, default=128)  # how many images are processed together
    # WHAT: Optional limit on training images.
    # HOW: 0 means use all 50,000.
    # WHY: Used only for tests.
    # PROS & CONS: + Fast tests. - Must stay 0 for results.
    # EXAMPLE: --max-train-samples 64
    p.add_argument("--max-train-samples", type=int, default=0, help="0 = full 50k training set")  # use only part of the training images (0 = use all)
    # WHAT: Helper processes for loading images.
    # HOW: Integer, default 2.
    # WHY: Colab has 2 CPU cores.
    # PROS & CONS: + Faster loading. - More processes use more RAM.
    # EXAMPLE: --num-workers 2
    p.add_argument("--num-workers", type=int, default=2)  # how many helper processes load images
    # WHAT: Folder, results folder and saved-model folder options.
    # HOW: Plain paths with defaults data, outputs and checkpoints.
    # WHY: Lets notebooks and tests use different folders.
    # PROS & CONS: + Flexible. - Mistakes can mix quick-test and real files.
    # EXAMPLE: --out-dir outputs_vit_rsfgsm
    p.add_argument("--data-root", default="data")  # folder for the dataset
    p.add_argument("--ckpt-dir", default=None)  # folder for saved models
    p.add_argument("--out-dir", default="outputs")  # folder for result tables and graphs
    # WHAT: Google Drive backup folder.
    # HOW: Optional path under /content/drive.
    # WHY: Colab disks are wiped at the end of a session.
    # PROS & CONS: + Lose nothing on a disconnect. - Needs Drive mounted.
    # EXAMPLE: --drive-dir /content/drive/MyDrive/adv_ckpts
    p.add_argument("--drive-dir", default=None, help="e.g. /content/drive/MyDrive/adv_ckpts (Colab)")  # Google Drive folder for backups
    # WHAT: Ignore existing adversarial checkpoints.
    # HOW: Boolean flag.
    # WHY: Needed when settings change.
    # PROS & CONS: + Explicit. - Deletes progress.
    # EXAMPLE: --force-retrain
    p.add_argument("--force-retrain", action="store_true", help="ignore existing adversarial checkpoints")  # train again even if saved models exist
    # WHAT: Upload progress to GitHub automatically.
    # HOW: Boolean flag; needs a token.
    # WHY: Backup of long runs.
    # PROS & CONS: + Remote copy. - Token handling must stay secret.
    # EXAMPLE: --auto-push
    p.add_argument("--auto-push", action="store_true", help="git commit+push after each stage / every 5 epochs")  # upload progress to GitHub automatically
    # WHAT: Reads and returns the typed options.
    # HOW: argparse parses sys.argv.
    # WHY: main() uses them as args.<name>.
    # PROS & CONS: + Standard. - None.
    # EXAMPLE: args.epochs == 10.
    return p.parse_args()  # read the typed options and give them back


# WHAT: Deletes old quick-test result files.
# HOW: Reads outputs/run_info.json; if it says mode == quick-test, removes the generated files.
# WHY: Quick-test numbers (64 images, 2 PGD steps) must never be mistaken for results.
# PROS & CONS: + Prevents wrong numbers in the report. - Deletes files, so only quick-test files are removed.
# EXAMPLE: '[outputs] removed stale quick-test artifacts'.
def clear_stale_quick_test_outputs(out_dir):  # deletes old quick-test results so fake numbers never mix with real ones
    """Quick-test artifacts are committed to git; delete them so they can never pass for full-run results."""
    # WHAT: Reads which kind of run produced the files.
    # HOW: Loads run_info.json (empty dict if missing).
    # WHY: Decides whether the files are quick-test output.
    # PROS & CONS: + Safe. - Relies on run_info.json being written.
    # EXAMPLE: {'mode': 'quick-test', ...}
    info = load_json(os.path.join(out_dir, "run_info.json"), {}) or {}  # read the note that says which kind of run made the files
    # WHAT: Stops unless the files are from a quick test.
    # HOW: Compares the stored mode.
    # WHY: Real results are never deleted.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: mode == 'full-run' returns early.
    if info.get("mode") != "quick-test":  # the files are from a real run (or missing)
        return  # keep them
    # WHAT: Deletes the known generated files.
    # HOW: Loop over names, os.remove if present.
    # WHY: Clean slate before a real run.
    # PROS & CONS: + Precise list. - Must be updated when new outputs are added.
    # EXAMPLE: outputs/results_summary.csv removed.
    for name in ("acc_vs_epsilon.png", "clean_vs_robust_tradeoff.png", "transferability_heatmap.png",  # the files made by a quick test
                 "adversarial_examples_grid.png", "results_summary.csv", "results_summary.md",  # more of those files
                 "transferability.csv", "run_info.json"):  # and the last ones
        p = os.path.join(out_dir, name)  # full path of the file
        if os.path.exists(p):  # the file exists
            os.remove(p)  # delete it
    # WHAT: Tells the user what happened.
    # HOW: Console message.
    # WHY: Transparency.
    # PROS & CONS: + Visible. - None.
    # EXAMPLE: [outputs] removed stale quick-test artifacts
    print("[outputs] removed stale quick-test artifacts")  # tell the user


# WHAT: The whole experiment in order.
# HOW: Load models, train what is missing, evaluate, save tables, draw figures, write a run summary.
# WHY: A single entry point makes the experiment repeatable.
# PROS & CONS: + Resumable; nothing existing is retrained. - Long function.
# EXAMPLE: python main.py --full-run took 111 minutes in total for the final ViT run.
def main():  # the main program: runs all the steps in order
    # WHAT: Reads the options and sets the mode.
    # HOW: quick is True for the smoke test.
    # WHY: Used throughout main().
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: quick = False for the real run.
    args = parse_args()  # read the typed options
    quick = args.quick_test  # True if this is the tiny test run
    # WHAT: Chooses GPU or CPU and fixes the random seed.
    # HOW: get_device and set_seed(42).
    # WHY: Speed and repeatability.
    # PROS & CONS: + Portable. - CPU is far too slow for a full run.
    # EXAMPLE: cuda on Colab.
    device = get_device()  # GPU if there is one, else CPU
    set_seed(SEED)  # fix the random numbers so results can be repeated
    # WHAT: Default settings of a real run.
    # HOW: PGD uses 10 steps, no batch limit, no note on the figures.
    # WHY: Quick tests override them below.
    # PROS & CONS: + Clear defaults. - None.
    # EXAMPLE: pgd_steps = 10.
    pgd_steps, max_batches, tag = 10, 0, ""  # normal settings: PGD uses 10 steps, no batch limit, no note on graphs
    # WHAT: Shrinks everything for the smoke test.
    # HOW: 1 epoch, 64 test images, batch 16, 2 PGD steps, 2 batches per epoch.
    # WHY: Runs in a few minutes on a CPU.
    # PROS & CONS: + Fast verification. - Numbers are invalid as results.
    # EXAMPLE: Plots get the note 'QUICK-TEST ... not a result'.
    if quick:  # tiny test run: make everything very small
        args.epochs, args.eval_samples, args.batch_size = 1, 64, 16  # 1 epoch, 64 test images, 16 images per batch
        args.max_train_samples, args.vit_clean_epochs, args.num_workers = 64, 1, 0  # 64 training images, 1 normal-training epoch, no helper processes
        args.drive_dir = None  # do not touch the real Google Drive backups
        pgd_steps, max_batches = 2, 2  # only 2 attack steps and 2 batches per epoch
        tag = "QUICK-TEST smoke run (tiny N, 1 epoch, 2 PGD steps) - not a result"  # note printed on the graphs
    # WHAT: Chooses and creates the folders.
    # HOW: Quick tests use checkpoints_quick_test so real checkpoints are never mixed with fake ones.
    # WHY: Protects real results.
    # PROS & CONS: + Safe. - Two folders to know.
    # EXAMPLE: checkpoints/ for real runs.
    args.ckpt_dir = args.ckpt_dir or ("checkpoints_quick_test" if quick else "checkpoints")  # saved models go to a separate folder for quick tests
    # WHAT: Creates the folders if missing.
    # HOW: makedirs with exist_ok.
    # WHY: Avoids errors.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: outputs/ appears on the first run.
    os.makedirs(args.ckpt_dir, exist_ok=True)  # create the saved-models folder
    os.makedirs(args.out_dir, exist_ok=True)  # create the results folder
    # WHAT: Converts eps to pixel units.
    # HOW: 8 becomes 8/255 = 0.0314.
    # WHY: Images are in [0, 1].
    # PROS & CONS: + Correct scale. - None.
    # EXAMPLE: 0.03137
    eps_train = args.eps / 255.0  # change e.g. 8 into 8/255, because pixel values are between 0 and 1

    # WHAT: Prints the setup.
    # HOW: Device, mode, checkpoint folder.
    # WHY: Shows what is about to run.
    # PROS & CONS: + Transparent. - None.
    # EXAMPLE: Device: cuda | mode: full-run | ckpt: checkpoints
    print(f"Device: {device} | mode: {'quick-test' if quick else 'full-run'} | ckpt: {args.ckpt_dir}")  # show the setup
    # WHAT: Warns about a full run on a CPU.
    # HOW: Console warning.
    # WHY: Would take days.
    # PROS & CONS: + Prevents wasted time. - Does not stop the run.
    # EXAMPLE: WARNING: full-run on CPU will be extremely slow.
    if device.type == "cpu" and not quick:  # a full run without a GPU
        print("WARNING: full-run on CPU will be extremely slow. Use Colab/Kaggle GPU (or --quick-test).")  # warn the user

    # WHAT: Restores saved models from Drive and clears old quick-test outputs.
    # HOW: Copies missing files back; deletes stale quick-test files in real runs.
    # WHY: A new Colab session can continue.
    # PROS & CONS: + No retraining. - Needs Drive.
    # EXAMPLE: cnn_adv.pth restored.
    restore_from_drive(args.ckpt_dir, args.drive_dir)  # bring back saved models from Google Drive (if used)
    # WHAT: Cleanup only for real runs.
    # HOW: Calls clear_stale_quick_test_outputs.
    # WHY: Real runs start from clean outputs.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: In a real run an old quick-test run_info.json triggers the deletion of the fake result files.
    if not quick:  # real run
        clear_stale_quick_test_outputs(args.out_dir)  # remove old quick-test results
    # WHAT: Defines the upload function.
    # HOW: A function that uploads, or does nothing without --auto-push.
    # WHY: Call sites stay the same either way.
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: push('results: evaluation tables and plots').
    push = (lambda msg: git_auto_push(msg)) if args.auto_push else (lambda msg: None)  # a function that uploads to GitHub, or does nothing if auto-push is off
    # WHAT: Starts the stopwatch.
    # HOW: Wall-clock time.
    # WHY: total_minutes is recorded in run_info.json.
    # PROS & CONS: + Cheap. - Includes download time (22 minutes once).
    # EXAMPLE: 111.05 minutes for the final run.
    t_start = time.time()  # note the start time

    # WHAT: Paths of the clean model files.
    # HOW: cnn_clean.pth and vit_clean.pth in the checkpoint folder.
    # WHY: Saved clean models make runs repeatable.
    # PROS & CONS: + Reuse. - None.
    # EXAMPLE: checkpoints/cnn_clean.pth
    cnn_path = os.path.join(args.ckpt_dir, "cnn_clean.pth")  # file of the normal CNN's weights
    vit_path = os.path.join(args.ckpt_dir, "vit_clean.pth")  # file of the normal ViT's weights

    # WHAT: Loads or creates the clean CNN.
    # HOW: If saved: build an empty ResNet-20 and fill it with the saved weights; otherwise download the
    #      pretrained CIFAR-10 weights and save them.
    # WHY: The same weights in every session.
    # PROS & CONS: + Repeatable. - The CNN was never trained from scratch by us.
    # EXAMPLE: '[CNN-Clean] loaded from checkpoint'.
    if os.path.exists(cnn_path):  # the CNN was already saved earlier
        cnn_clean = load_checkpoint(load_cnn_clean(device, pretrained=False), cnn_path, device)  # build an empty CNN and fill it with the saved weights
        print("[CNN-Clean] loaded from checkpoint")  # tell the user
    else:  # first run
        cnn_clean = load_cnn_clean(device)  # download the ready-made CNN
        save_checkpoint(cnn_clean, cnn_path)  # save it for next time

    # WHAT: Loads or creates the clean ViT.
    # HOW: If saved: load. Otherwise download ImageNet weights and remember it needs clean fine-tuning.
    # WHY: A new classifier head has random weights.
    # PROS & CONS: + Skips repeated fine-tuning. - A lost checkpoint means a second fine-tune (97.4% vs 97.1%).
    # EXAMPLE: '[ViT-Clean] loaded from checkpoint'.
    vit_needs_ft = False  # does the ViT still need normal training? assume no
    # WHAT: Loads the saved clean ViT when available.
    # HOW: Builds an untrained shell and loads the saved weights.
    # WHY: Avoids repeating the 5-epoch fine-tune.
    # PROS & CONS: + Saves time. - Needs the same architecture.
    # EXAMPLE: vit_clean.pth.
    if os.path.exists(vit_path):  # the ViT was already saved earlier
        shell, _ = load_vit_clean(device, args.vit_backbone, pretrained=False)  # build an empty ViT
        vit_clean = load_checkpoint(shell, vit_path, device)  # fill it with the saved weights
        print("[ViT-Clean] loaded from checkpoint")  # tell the user
    else:  # first run
        vit_clean, vit_needs_ft = load_vit_clean(device, args.vit_backbone)  # download the ViT and learn if it needs normal training

    # WHAT: Downloads CIFAR-10 if needed.
    # HOW: Calls prepare_data.
    # WHY: Training needs the data.
    # PROS & CONS: + Cached. - First download is slow.
    # EXAMPLE: data/cifar-10-batches-py.
    prepare_data(args.data_root, args.drive_dir)  # make sure the dataset is downloaded
    # WHAT: Builds the training loader.
    # HOW: Shuffled batches of 128 with augmentation.
    # WHY: Used by both clean and adversarial training.
    # PROS & CONS: + Standard. - None.
    # EXAMPLE: 390 batches per epoch.
    train_loader = get_train_loader(args.batch_size, args.data_root, args.max_train_samples,  # the object that hands out training images in groups
                                    num_workers=args.num_workers)  # number of helper processes

    # WHAT: Clean fine-tuning of the ViT when needed.
    # HOW: 5 epochs of normal training, then save.
    # WHY: The ViT must classify CIFAR-10 before it is attacked.
    # PROS & CONS: + 97.1% clean accuracy. - Extra time.
    # EXAMPLE: vit_clean.pth saved.
    if vit_needs_ft:  # the new ViT must first learn CIFAR-10 with normal training
        print(f"[ViT-Clean] clean fine-tune on CIFAR-10 for {args.vit_clean_epochs} epochs")  # tell the user
        clean_finetune(vit_clean, train_loader, epochs=args.vit_clean_epochs, device=device,  # normal training, no tricked images
                       max_batches=max_batches)  # batch limit (only used by the quick test)
        save_checkpoint(vit_clean, vit_path)  # save the trained ViT
    # WHAT: Backs up the clean models.
    # HOW: Drive copy and optional GitHub push.
    # WHY: Safety.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: cnn_clean.pth and vit_clean.pth are copied to the Drive folder after they are created.
    sync_to_drive(args.ckpt_dir, args.drive_dir)  # back up the saved models to Google Drive (if used)
    push("chore(train): clean CNN/ViT baseline checkpoints")  # upload to GitHub (if switched on)

    # WHAT: Path of the training log.
    # HOW: JSON file in the checkpoint folder.
    # WHY: Training history for the report.
    # PROS & CONS: + Persistent. - None.
    # EXAMPLE: checkpoints/training_history.json
    hist_path = os.path.join(args.ckpt_dir, "training_history.json")  # the log file of training progress
    # WHAT: Will hold the two adversarially trained models.
    # HOW: Empty dictionary filled in the loop.
    # WHY: Evaluation needs all four models.
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: {}
    adv_models = {}  # will hold the two defended models
    # WHAT: Loops over the CNN and the ViT.
    # HOW: (file tag, clean model, training attack, display name).
    # WHY: Both models get exactly the same recipe.
    # PROS & CONS: + Fair. - None.
    # EXAMPLE: ('vit_adv', vit_clean, 'pgd5', 'ViT-AdvTrained').
    for tag_name, clean, method, label in (("cnn_adv", cnn_clean, args.cnn_method, "CNN-AdvTrained"),  # first: (file name, normal model, attack to train with, display name) for the CNN
                                           ("vit_adv", vit_clean, args.vit_method, "ViT-AdvTrained")):  # second: the same four things for the ViT
        # WHAT: File of the defended model.
        # HOW: Path of the .pth file.
        # WHY: Used for skip/resume.
        # PROS & CONS: + Simple. - None.
        # EXAMPLE: checkpoints/vit_adv.pth
        ckpt = os.path.join(args.ckpt_dir, f"{tag_name}.pth")  # file of the defended model's weights
        # WHAT: Copy of the clean model to train.
        # HOW: deepcopy duplicates all weights.
        # WHY: The clean model must stay unchanged so that before/after can be compared.
        # PROS & CONS: + Clean baseline preserved. - Doubles model memory briefly.
        # EXAMPLE: ViT-Clean stays at 97.1%.
        model = copy.deepcopy(clean)  # make a full separate copy, so the normal model stays unchanged
        # WHAT: Skips training if a finished checkpoint exists.
        # HOW: Loads the weights.
        # WHY: Never repeats work.
        # PROS & CONS: + Saves up to 75 minutes. - Trusts the log file.
        # EXAMPLE: '[ViT-AdvTrained] completed checkpoint found -> skipping training'.
        if (os.path.exists(ckpt) and not args.force_retrain  # a saved defended model exists and we were not told to retrain
                and is_training_complete(hist_path, tag_name, args.epochs)):  # and the log says its training finished
            load_checkpoint(model, ckpt, device)  # just load it
            print(f"[{label}] completed checkpoint found -> skipping training")  # tell the user
        # WHAT: Trains the defended model.
        # HOW: Prints the settings, optionally deletes old files, runs adversarial_finetune.
        # WHY: Defence training.
        # PROS & CONS: + Resumable. - Long run.
        # EXAMPLE: '[ViT-AdvTrained] adversarial fine-tuning: 10 epochs, pgd5, eps=8.0/255'.
        else:  # we need to train it
            print(f"[{label}] adversarial fine-tuning: {args.epochs} epochs, {method}, eps={args.eps}/255")  # tell the user the settings
            # WHAT: Deletes old files when retraining is forced.
            # HOW: os.remove of the weights and resume state.
            # WHY: A clean restart.
            # PROS & CONS: + Explicit. - Irreversible.
            # EXAMPLE: --force-retrain.
            if args.force_retrain:  # start again from zero
                for f in (ckpt, os.path.join(args.ckpt_dir, f"{tag_name}_state.pt")):  # the saved model and the "continue later" file
                    if os.path.exists(f):  # the file exists
                        os.remove(f)  # delete it
            # WHAT: The actual defence training.
            # HOW: 10 epochs, PGD-5, mixed precision, checkpointing every epoch.
            # WHY: Main experiment.
            # PROS & CONS: + Resumable. - Time-consuming.
            # EXAMPLE: CNN 6.3 min, ViT 74.7 min on a T4.
            adversarial_finetune(model, train_loader, epochs=args.epochs, eps=eps_train, method=method,  # train the copy using normal AND tricked images
                                 lr=args.lr, device=device, tag=tag_name, ckpt_dir=args.ckpt_dir,  # learning rate, GPU/CPU, name, folder for saved files
                                 drive_dir=args.drive_dir, auto_push_every=5 if args.auto_push else 0,  # Drive backup folder; upload to GitHub every 5 epochs if switched on
                                 max_batches=max_batches)  # batch limit (only used by the quick test)
            # WHAT: Uploads progress after training.
            # HOW: GitHub push (if enabled).
            # WHY: Remote backup.
            # PROS & CONS: + Safe. - None.
            # EXAMPLE: Commit message: 'chore(train): ViT-AdvTrained 10-epoch adversarial fine-tuning finished'.
            push(f"chore(train): {label} {args.epochs}-epoch adversarial fine-tuning finished")  # upload to GitHub (if switched on)
        # WHAT: Stores the defended model in test mode.
        # HOW: eval().
        # WHY: Ready for evaluation.
        # PROS & CONS: + Clear. - None.
        # EXAMPLE: adv_models['CNN-AdvTrained'].
        adv_models[label] = model.eval()  # keep this defended model, in test mode

    # WHAT: All four models together.
    # HOW: Two clean, two defended.
    # WHY: The core comparison.
    # PROS & CONS: + Clear. - None.
    # EXAMPLE: dict with 4 entries.
    models = {"CNN-Clean": cnn_clean, "ViT-Clean": vit_clean, **adv_models}  # all four models together: 2 normal + 2 defended
    # WHAT: Freezes every weight.
    # HOW: requires_grad_(False).
    # WHY: Attacks need gradients only for the image; saves memory and time.
    # PROS & CONS: + Faster attacks. - Models can no longer be trained.
    # EXAMPLE: No weights change during evaluation.
    for m in models.values():  # go through the four models
        for prm in m.parameters():  # go through every learnable number in the model
            prm.requires_grad_(False)  # freeze them: attacks only need to change the image, not the model (saves memory and time)

    # WHAT: Builds the fixed evaluation subset.
    # HOW: 1000 stratified test images, seed 42.
    # WHY: Every model sees the same images.
    # PROS & CONS: + Fair. - Subset only.
    # EXAMPLE: (1000, 3, 32, 32).
    X, Y = get_eval_tensors(args.eval_samples, args.data_root)  # the fixed test images X and their true labels Y
    print(f"Evaluation subset: {len(X)} stratified test images (seed {SEED})")  # tell the user how many test images are used
    # WHAT: Accuracy versus epsilon.
    # HOW: FGSM and PGD-10 at eps 0, 1, 2, 4, 8, 16 for all four models.
    # WHY: The mentor's required curves.
    # PROS & CONS: + Complete. - Slow for the ViT.
    # EXAMPLE: 48 rows.
    sweep = evaluate_epsilon_sweep(models, X, Y, device, pgd_steps=pgd_steps)  # test every model against every attack at every strength
    # WHAT: Transfer matrix at eps = 8/255.
    # HOW: PGD-10 crafted on each model, tested on each.
    # WHY: Black-box view.
    # PROS & CONS: + Extra evidence. - Only one eps.
    # EXAMPLE: 4x4 table.
    transfer = evaluate_transferability(models, X, Y, device, eps=8 / 255, pgd_steps=pgd_steps)  # test whether tricks made on one model fool the others
    # WHAT: Note text for the Markdown report.
    # HOW: Marks quick-test output as invalid.
    # WHY: Honesty about what the numbers are.
    # PROS & CONS: + Clear. - None.
    # EXAMPLE: N=1000 stratified CIFAR-10 test images, seed 42, PGD-10.
    note = (f"Quick-test smoke run only (N={len(X)}, {pgd_steps} PGD steps): not valid results."  # note text for a quick test
            if quick else f"N={len(X)} stratified CIFAR-10 test images, seed {SEED}, PGD-{pgd_steps}.")  # note text for a real run
    # WHAT: Writes the tables to files.
    # HOW: CSV and Markdown.
    # WHY: Tracked in git.
    # PROS & CONS: + Visible on GitHub. - None.
    # EXAMPLE: outputs/results_summary.csv
    export_results(sweep, transfer, args.out_dir, note)  # save the result tables as files

    # WHAT: Draws the four figures.
    # HOW: make_all_plots.
    # WHY: Visual results.
    # PROS & CONS: + Report-ready. - None.
    # EXAMPLE: outputs/*.png.
    make_all_plots(sweep, transfer, models, X, Y, device, args.out_dir, pgd_steps=pgd_steps, tag=tag)  # draw and save the four graphs

    # WHAT: Summary of the run.
    # HOW: Mode, device, epochs, settings, clean accuracy of each model, total minutes.
    # WHY: Documents provenance of the results.
    # PROS & CONS: + Traceable. - None.
    # EXAMPLE: outputs/run_info.json.
    info = {"mode": "quick-test" if quick else "full-run", "device": str(device), "epochs": args.epochs,  # a summary of this run: kind of run, device, epochs
            "eval_samples": len(X), "pgd_eval_steps": pgd_steps, "cnn_method": args.cnn_method,  # number of test images, attack steps, CNN training attack
            "vit_method": args.vit_method, "vit_backbone": args.vit_backbone,  # ViT training attack, which ViT
            "clean_acc_full_subset": {k: accuracy(m, X, Y, device) for k, m in models.items()},  # normal accuracy of each model
            "total_minutes": round((time.time() - t_start) / 60, 2)}  # total time in minutes
    # WHAT: Saves the summary.
    # HOW: JSON.
    # WHY: Used by the clean-up check and the notebook.
    # PROS & CONS: + Persistent. - None.
    # EXAMPLE: run_info.json.
    save_json(info, os.path.join(args.out_dir, "run_info.json"))  # save the summary to a file
    # WHAT: Shows the summary.
    # HOW: indent=2.
    # WHY: Visible in Colab.
    # PROS & CONS: + Visible. - None.
    # EXAMPLE: Prints the run_info dictionary, e.g. {'mode': 'full-run', 'epochs': 10, ...}.
    print(json.dumps(info, indent=2))  # show the summary on screen
    # WHAT: Uploads results.
    # HOW: GitHub push (if enabled).
    # WHY: Final backup.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: Commits outputs/*.csv, *.md and *.png with the message 'results: evaluation tables and plots'.
    push("results: evaluation tables and plots")  # upload results to GitHub (if switched on)
    # WHAT: Final message.
    # HOW: Run time.
    # WHY: Closure.
    # PROS & CONS: + Clear. - None.
    # EXAMPLE: Done in 111.05 min. Outputs in outputs/
    print(f"Done in {info['total_minutes']} min. Outputs in {args.out_dir}/")  # tell the user it is finished


# WHAT: Starts main() when the file is run.
# HOW: Python sets __name__ to '__main__'.
# WHY: Importable without side effects.
# PROS & CONS: + Standard. - None.
# EXAMPLE: python main.py
if __name__ == "__main__":  # true only when this file is run directly (python main.py)
    main()  # start the program
