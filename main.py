"""CLI entry point: CNN vs ViT adversarial robustness on CIFAR-10.

  python main.py --quick-test            # CPU smoke test (64 test images, 1 tiny epoch), no GPU needed
  python main.py --full-run              # Colab/Kaggle GPU: 10-epoch adversarial fine-tuning + full evaluation
  python main.py --epochs 10 --eval-samples 1000 --auto-push

Checkpoints in checkpoints/ are reused automatically: nothing that already exists is retrained.
"""
import argparse  # read options typed in the terminal
import copy  # make independent copies of models
import json  # print / save results as JSON text
import os  # work with file and folder paths
import time  # measure how long the run takes

from src.data import get_eval_tensors, get_train_loader, prepare_data  # functions that get the pictures
from src.defense import adversarial_finetune, clean_finetune, is_training_complete  # functions that train the models
from src.evaluate import accuracy, evaluate_epsilon_sweep, evaluate_transferability, export_results  # functions that test the models and save results
from src.models import load_checkpoint, load_cnn_clean, load_vit_clean, save_checkpoint  # functions that create, load and save models
from src.utils import (SEED, get_device, git_auto_push, load_json, restore_from_drive, save_json, set_seed,  # small helper tools
                       sync_to_drive)  # backup to Google Drive
from src.visualize import make_all_plots  # function that draws the graphs


def parse_args():  # reads the options typed after "python main.py", e.g. --epochs 5
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # the option reader; shows the text at the top of this file when asked for help
    mode = p.add_mutually_exclusive_group()  # options in this group cannot be used together
    mode.add_argument("--quick-test", action="store_true", help="CPU smoke test: 64 images, 1 short epoch")  # tiny fast run just to check the code works
    mode.add_argument("--full-run", action="store_true", help="Full GPU run (default when no mode is given)")  # the real run
    p.add_argument("--epochs", type=int, default=10, help="adversarial fine-tuning epochs")  # how many times the model sees the whole training set
    p.add_argument("--eval-samples", type=int, default=1000, help="stratified test subset size")  # how many test images are used for testing
    p.add_argument("--eps", type=float, default=8.0, help="training epsilon in /255 units")  # biggest change per pixel used while training the defence
    p.add_argument("--cnn-method", choices=["pgd5", "rs_fgsm"], default="pgd5")  # which attack trains the CNN's defence
    p.add_argument("--vit-method", choices=["pgd5", "rs_fgsm"], default="pgd5",  # which attack trains the ViT's defence
                   help="pgd5 matches the CNN (fair comparison). rs_fgsm is ~2.5x faster but the ViT showed "  # help text, part 1
                        "signs of catastrophic overfitting with it (see outputs_vit_rsfgsm/)")  # help text, part 2
    p.add_argument("--vit-backbone", choices=["timm-tiny", "hf-base"], default="timm-tiny")  # which ViT to use: small (fast) or big (slow)
    p.add_argument("--vit-clean-epochs", type=int, default=5, help="clean fine-tune epochs for timm ViT-Tiny")  # epochs of normal training for the small ViT
    p.add_argument("--lr", type=float, default=1e-4)  # learning rate = how big each weight update is
    p.add_argument("--batch-size", type=int, default=128)  # how many images are processed together
    p.add_argument("--max-train-samples", type=int, default=0, help="0 = full 50k training set")  # use only part of the training images (0 = use all)
    p.add_argument("--num-workers", type=int, default=2)  # how many helper processes load images
    p.add_argument("--data-root", default="data")  # folder for the dataset
    p.add_argument("--ckpt-dir", default=None)  # folder for saved models
    p.add_argument("--out-dir", default="outputs")  # folder for result tables and graphs
    p.add_argument("--drive-dir", default=None, help="e.g. /content/drive/MyDrive/adv_ckpts (Colab)")  # Google Drive folder for backups
    p.add_argument("--force-retrain", action="store_true", help="ignore existing adversarial checkpoints")  # train again even if saved models exist
    p.add_argument("--auto-push", action="store_true", help="git commit+push after each stage / every 5 epochs")  # upload progress to GitHub automatically
    return p.parse_args()  # read the typed options and give them back


def clear_stale_quick_test_outputs(out_dir):  # deletes old quick-test results so fake numbers never mix with real ones
    """Quick-test artifacts are committed to git; delete them so they can never pass for full-run results."""
    info = load_json(os.path.join(out_dir, "run_info.json"), {}) or {}  # read the note that says which kind of run made the files
    if info.get("mode") != "quick-test":  # the files are from a real run (or missing)
        return  # keep them
    for name in ("acc_vs_epsilon.png", "clean_vs_robust_tradeoff.png", "transferability_heatmap.png",  # the files made by a quick test
                 "adversarial_examples_grid.png", "results_summary.csv", "results_summary.md",  # more of those files
                 "transferability.csv", "run_info.json"):  # and the last ones
        p = os.path.join(out_dir, name)  # full path of the file
        if os.path.exists(p):  # the file exists
            os.remove(p)  # delete it
    print("[outputs] removed stale quick-test artifacts")  # tell the user


def main():  # the main program: runs all the steps in order
    args = parse_args()  # read the typed options
    quick = args.quick_test  # True if this is the tiny test run
    device = get_device()  # GPU if there is one, else CPU
    set_seed(SEED)  # fix the random numbers so results can be repeated
    pgd_steps, max_batches, tag = 10, 0, ""  # normal settings: PGD uses 10 steps, no batch limit, no note on graphs
    if quick:  # tiny test run: make everything very small
        args.epochs, args.eval_samples, args.batch_size = 1, 64, 16  # 1 epoch, 64 test images, 16 images per batch
        args.max_train_samples, args.vit_clean_epochs, args.num_workers = 64, 1, 0  # 64 training images, 1 normal-training epoch, no helper processes
        args.drive_dir = None  # do not touch the real Google Drive backups
        pgd_steps, max_batches = 2, 2  # only 2 attack steps and 2 batches per epoch
        tag = "QUICK-TEST smoke run (tiny N, 1 epoch, 2 PGD steps) - not a result"  # note printed on the graphs
    args.ckpt_dir = args.ckpt_dir or ("checkpoints_quick_test" if quick else "checkpoints")  # saved models go to a separate folder for quick tests
    os.makedirs(args.ckpt_dir, exist_ok=True)  # create the saved-models folder
    os.makedirs(args.out_dir, exist_ok=True)  # create the results folder
    eps_train = args.eps / 255.0  # change e.g. 8 into 8/255, because pixel values are between 0 and 1

    print(f"Device: {device} | mode: {'quick-test' if quick else 'full-run'} | ckpt: {args.ckpt_dir}")  # show the setup
    if device.type == "cpu" and not quick:  # a full run without a GPU
        print("WARNING: full-run on CPU will be extremely slow. Use Colab/Kaggle GPU (or --quick-test).")  # warn the user

    restore_from_drive(args.ckpt_dir, args.drive_dir)  # bring back saved models from Google Drive (if used)
    if not quick:  # real run
        clear_stale_quick_test_outputs(args.out_dir)  # remove old quick-test results
    push = (lambda msg: git_auto_push(msg)) if args.auto_push else (lambda msg: None)  # a function that uploads to GitHub, or does nothing if auto-push is off
    t_start = time.time()  # note the start time

    cnn_path = os.path.join(args.ckpt_dir, "cnn_clean.pth")  # file of the normal CNN's weights
    vit_path = os.path.join(args.ckpt_dir, "vit_clean.pth")  # file of the normal ViT's weights

    if os.path.exists(cnn_path):  # the CNN was already saved earlier
        cnn_clean = load_checkpoint(load_cnn_clean(device, pretrained=False), cnn_path, device)  # build an empty CNN and fill it with the saved weights
        print("[CNN-Clean] loaded from checkpoint")  # tell the user
    else:  # first run
        cnn_clean = load_cnn_clean(device)  # download the ready-made CNN
        save_checkpoint(cnn_clean, cnn_path)  # save it for next time

    vit_needs_ft = False  # does the ViT still need normal training? assume no
    if os.path.exists(vit_path):  # the ViT was already saved earlier
        shell, _ = load_vit_clean(device, args.vit_backbone, pretrained=False)  # build an empty ViT
        vit_clean = load_checkpoint(shell, vit_path, device)  # fill it with the saved weights
        print("[ViT-Clean] loaded from checkpoint")  # tell the user
    else:  # first run
        vit_clean, vit_needs_ft = load_vit_clean(device, args.vit_backbone)  # download the ViT and learn if it needs normal training

    prepare_data(args.data_root, args.drive_dir)  # make sure the dataset is downloaded
    train_loader = get_train_loader(args.batch_size, args.data_root, args.max_train_samples,  # the object that hands out training images in groups
                                    num_workers=args.num_workers)  # number of helper processes

    if vit_needs_ft:  # the new ViT must first learn CIFAR-10 with normal training
        print(f"[ViT-Clean] clean fine-tune on CIFAR-10 for {args.vit_clean_epochs} epochs")  # tell the user
        clean_finetune(vit_clean, train_loader, epochs=args.vit_clean_epochs, device=device,  # normal training, no tricked images
                       max_batches=max_batches)  # batch limit (only used by the quick test)
        save_checkpoint(vit_clean, vit_path)  # save the trained ViT
    sync_to_drive(args.ckpt_dir, args.drive_dir)  # back up the saved models to Google Drive (if used)
    push("chore(train): clean CNN/ViT baseline checkpoints")  # upload to GitHub (if switched on)

    hist_path = os.path.join(args.ckpt_dir, "training_history.json")  # the log file of training progress
    adv_models = {}  # will hold the two defended models
    for tag_name, clean, method, label in (("cnn_adv", cnn_clean, args.cnn_method, "CNN-AdvTrained"),  # first: (file name, normal model, attack to train with, display name) for the CNN
                                           ("vit_adv", vit_clean, args.vit_method, "ViT-AdvTrained")):  # second: the same four things for the ViT
        ckpt = os.path.join(args.ckpt_dir, f"{tag_name}.pth")  # file of the defended model's weights
        model = copy.deepcopy(clean)  # make a full separate copy, so the normal model stays unchanged
        if (os.path.exists(ckpt) and not args.force_retrain  # a saved defended model exists and we were not told to retrain
                and is_training_complete(hist_path, tag_name, args.epochs)):  # and the log says its training finished
            load_checkpoint(model, ckpt, device)  # just load it
            print(f"[{label}] completed checkpoint found -> skipping training")  # tell the user
        else:  # we need to train it
            print(f"[{label}] adversarial fine-tuning: {args.epochs} epochs, {method}, eps={args.eps}/255")  # tell the user the settings
            if args.force_retrain:  # start again from zero
                for f in (ckpt, os.path.join(args.ckpt_dir, f"{tag_name}_state.pt")):  # the saved model and the "continue later" file
                    if os.path.exists(f):  # the file exists
                        os.remove(f)  # delete it
            adversarial_finetune(model, train_loader, epochs=args.epochs, eps=eps_train, method=method,  # train the copy using normal AND tricked images
                                 lr=args.lr, device=device, tag=tag_name, ckpt_dir=args.ckpt_dir,  # learning rate, GPU/CPU, name, folder for saved files
                                 drive_dir=args.drive_dir, auto_push_every=5 if args.auto_push else 0,  # Drive backup folder; upload to GitHub every 5 epochs if switched on
                                 max_batches=max_batches)  # batch limit (only used by the quick test)
            push(f"chore(train): {label} {args.epochs}-epoch adversarial fine-tuning finished")  # upload to GitHub (if switched on)
        adv_models[label] = model.eval()  # keep this defended model, in test mode

    models = {"CNN-Clean": cnn_clean, "ViT-Clean": vit_clean, **adv_models}  # all four models together: 2 normal + 2 defended
    for m in models.values():  # go through the four models
        for prm in m.parameters():  # go through every learnable number in the model
            prm.requires_grad_(False)  # freeze them: attacks only need to change the image, not the model (saves memory and time)

    X, Y = get_eval_tensors(args.eval_samples, args.data_root)  # the fixed test images X and their true labels Y
    print(f"Evaluation subset: {len(X)} stratified test images (seed {SEED})")  # tell the user how many test images are used
    sweep = evaluate_epsilon_sweep(models, X, Y, device, pgd_steps=pgd_steps)  # test every model against every attack at every strength
    transfer = evaluate_transferability(models, X, Y, device, eps=8 / 255, pgd_steps=pgd_steps)  # test whether tricks made on one model fool the others
    note = (f"Quick-test smoke run only (N={len(X)}, {pgd_steps} PGD steps): not valid results."  # note text for a quick test
            if quick else f"N={len(X)} stratified CIFAR-10 test images, seed {SEED}, PGD-{pgd_steps}.")  # note text for a real run
    export_results(sweep, transfer, args.out_dir, note)  # save the result tables as files

    make_all_plots(sweep, transfer, models, X, Y, device, args.out_dir, pgd_steps=pgd_steps, tag=tag)  # draw and save the four graphs

    info = {"mode": "quick-test" if quick else "full-run", "device": str(device), "epochs": args.epochs,  # a summary of this run: kind of run, device, epochs
            "eval_samples": len(X), "pgd_eval_steps": pgd_steps, "cnn_method": args.cnn_method,  # number of test images, attack steps, CNN training attack
            "vit_method": args.vit_method, "vit_backbone": args.vit_backbone,  # ViT training attack, which ViT
            "clean_acc_full_subset": {k: accuracy(m, X, Y, device) for k, m in models.items()},  # normal accuracy of each model
            "total_minutes": round((time.time() - t_start) / 60, 2)}  # total time in minutes
    save_json(info, os.path.join(args.out_dir, "run_info.json"))  # save the summary to a file
    print(json.dumps(info, indent=2))  # show the summary on screen
    push("results: evaluation tables and plots")  # upload results to GitHub (if switched on)
    print(f"Done in {info['total_minutes']} min. Outputs in {args.out_dir}/")  # tell the user it is finished


if __name__ == "__main__":  # true only when this file is run directly (python main.py)
    main()  # start the program
