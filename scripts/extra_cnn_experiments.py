"""Two extra CNN experiments that make the CNN-vs-ViT comparison stronger.

  A) ResNet-20 trained FROM SCRATCH on CIFAR-10 (the mentor's "train ResNet"), then adversarially
     fine-tuned with the same recipe as before (10 epochs, PGD-5, eps = 8/255).
  B) ResNet-18, ImageNet-pretrained, 224x224 input: the same input size and pretraining as the
     ViT-Tiny and a closer size (11M vs 5.7M weights). Clean fine-tune (5 epochs), then the same
     adversarial recipe.

For every new model it measures clean / FGSM / PGD-10 accuracy for eps in {0,1,2,4,8,16}/255 and the
stronger PGD-50 x5 restarts check at eps = 8/255 (same 1000 stratified test images as the report).

Run from the repository root on a GPU (Colab T4):
    python scripts/extra_cnn_experiments.py --quick          # ~2-5 min smoke test, run this first
    python scripts/extra_cnn_experiments.py --only resnet20  # ~25 min
    python scripts/extra_cnn_experiments.py --only resnet18  # ~1-1.5 h
    python scripts/extra_cnn_experiments.py                  # both, one after the other
Use --drive-dir /content/drive/MyDrive/extra_ckpts so a Colab disconnect does not lose progress
(training resumes automatically). Results go to outputs_extra_cnn/.
"""
# WHAT: Imports command-line, file, table and PyTorch libraries.
# HOW: pandas writes the CSV files; torch.nn.functional provides the loss.
# WHY: The script trains and evaluates extra models.
# PROS & CONS: + Standard. - None.
# EXAMPLE: python scripts/extra_cnn_experiments.py --quick
import argparse
import copy
import os
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F

# WHAT: Makes src/ and the sibling script importable.
# HOW: Inserts the repository root and the scripts folder into sys.path.
# WHY: The script lives in scripts/.
# PROS & CONS: + Works from any folder. - Path hack.
# EXAMPLE: from strong_attack_check import robust_accuracy.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repository root, so "src" can be imported
sys.path.insert(0, HERE)  # this folder, so strong_attack_check can be imported

# WHAT: Imports the project's building blocks.
# HOW: Data, training, evaluation, models, utilities and the strong attack.
# WHY: Same code as the main experiment, so the results are comparable.
# PROS & CONS: + Consistent. - None.
# EXAMPLE: adversarial_finetune, evaluate_epsilon_sweep.
from src.data import get_eval_tensors, get_train_loader, prepare_data  # noqa: E402
from src.defense import _autocast, adversarial_finetune, clean_finetune, is_training_complete  # noqa: E402
from src.evaluate import accuracy, evaluate_epsilon_sweep  # noqa: E402
from src.models import load_checkpoint, load_cnn_clean, load_resnet18_clean, save_checkpoint  # noqa: E402
from src.utils import SEED, get_device, restore_from_drive, save_json, set_seed, sync_to_drive  # noqa: E402
from strong_attack_check import robust_accuracy  # noqa: E402

# WHAT: Training budget.
# HOW: 8/255.
# WHY: Same as the main experiments.
# PROS & CONS: + Comparable. - None.
# EXAMPLE: 0.03137
EPS_TRAIN = 8 / 255  # training budget, same as the main experiments


# WHAT: Normal training of a randomly initialised model.
# HOW: SGD with momentum and a one-cycle learning-rate schedule, mixed precision.
# WHY: The mentor asked to 'train ResNet'; this trains ResNet-20 itself.
# PROS & CONS: + Standard CIFAR recipe, about 91-92% accuracy for ResNet-20. - Slower than using pretrained
#              weights.
# EXAMPLE: 100 epochs.
def train_from_scratch(model, loader, epochs, device, max_batches=0, lr=0.1):
    """Normal (clean) training of a randomly initialised model: SGD + one-cycle learning-rate schedule."""
    # WHAT: Mixed precision flag.
    # HOW: True on CUDA.
    # WHY: Speed.
    # PROS & CONS: + Faster. - None.
    # EXAMPLE: True on Colab.
    use_amp = device.type == "cuda"
    # WHAT: The optimiser.
    # HOW: SGD, learning rate 0.1, momentum 0.9, Nesterov, weight decay 5e-4.
    # WHY: The standard recipe for CIFAR ResNets.
    # PROS & CONS: + Proven. - Needs the schedule below.
    # EXAMPLE: lr peaks at 0.1.
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # WHAT: Number of steps per epoch.
    # HOW: Number of batches (limited in quick tests).
    # WHY: OneCycleLR needs the exact total number of steps.
    # PROS & CONS: + Exact. - None.
    # EXAMPLE: 390.
    per_epoch = min(len(loader), max_batches) if max_batches else len(loader)
    # WHAT: Learning-rate schedule.
    # HOW: Rises for 15% of the steps, then decays to near 0.
    # WHY: Fast and stable convergence for short training.
    # PROS & CONS: + Few epochs needed. - Changes every batch.
    # EXAMPLE: peak at step 5850 of 39000.
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * per_epoch, pct_start=0.15)
    # WHAT: Protects small fp16 gradients.
    # HOW: Scales the loss.
    # WHY: Stable mixed precision.
    # PROS & CONS: + Standard. - None.
    # EXAMPLE: scale 65536.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # WHAT: Epoch loop.
    # HOW: One full pass over the data per epoch.
    # WHY: Standard.
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: 100 epochs.
    for epoch in range(epochs):
        # WHAT: Training mode.
        # HOW: BatchNorm updates statistics.
        # WHY: Required for learning.
        # PROS & CONS: + Standard. - None.
        # EXAMPLE: model.training is True.
        model.train()
        # WHAT: Running totals.
        # HOW: Loss, correct predictions, image count.
        # WHY: Epoch statistics.
        # PROS & CONS: + Cheap. - None.
        # EXAMPLE: 0
        tot = correct = n = 0
        t0 = time.time()
        # WHAT: Loop over batches.
        # HOW: x images, y labels.
        # WHY: One update per batch.
        # PROS & CONS: + Standard. - None.
        # EXAMPLE: 390 batches.
        for b, (x, y) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            # WHAT: Forward pass in mixed precision.
            # HOW: Model output in fp16 on a GPU.
            # WHY: Speed.
            # PROS & CONS: + Faster. - Slight noise.
            # EXAMPLE: (128, 10) scores.
            with _autocast(device, use_amp):
                logits = model(x)
            # WHAT: The loss.
            # HOW: Cross-entropy in fp32.
            # WHY: Standard classification loss.
            # PROS & CONS: + Standard. - None.
            # EXAMPLE: About 2.3 at the start.
            loss = F.cross_entropy(logits.float(), y)
            # WHAT: Backward pass and weight update.
            # HOW: Scaled backward, optimiser step, scaler update, schedule step.
            # WHY: Learning.
            # PROS & CONS: + Standard. - None.
            # EXAMPLE: Weights change.
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            # WHAT: Updates the statistics.
            # HOW: Adds loss and correct counts.
            # WHY: Progress report.
            # PROS & CONS: + Cheap. - None.
            # EXAMPLE: train acc.
            tot += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        # WHAT: Prints every 10th epoch.
        # HOW: Also the first and last epoch.
        # WHY: Short logs.
        # PROS & CONS: + Readable. - None.
        # EXAMPLE: [scratch] epoch 10/100 | loss ... | train acc ...%
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"[scratch] epoch {epoch + 1}/{epochs} | loss {tot / max(n, 1):.4f} | "
                  f"train acc {100 * correct / max(n, 1):.2f}% | {time.time() - t0:.0f}s/epoch", flush=True)
    # WHAT: Returns the trained model in test mode.
    # HOW: eval().
    # WHY: Ready to use.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: scratch = train_from_scratch(...) returns the trained model with model.training == False.
    return model.eval()


# WHAT: Adversarial fine-tuning with the same recipe as the main experiments.
# HOW: Skips training if a finished checkpoint exists; otherwise PGD-5, 10 epochs, eps 8/255, lr 1e-4.
# WHY: Same recipe makes the new models comparable with the CNN and ViT of the report.
# PROS & CONS: + Resumable and consistent. - None.
# EXAMPLE: adv_stage(clean, 'resnet18_adv', 'ResNet18-AdvTrained', args, loader, hist_path).
def adv_stage(clean, tag, label, args, loader, hist_path):
    """Adversarial fine-tuning with the same recipe as the main experiments (skips if already finished)."""
    # WHAT: Checkpoint path.
    # HOW: Folder plus tag.
    # WHY: Used to skip finished training.
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: checkpoints_extra/resnet18_adv.pth
    ckpt = os.path.join(args.ckpt_dir, f"{tag}.pth")
    # WHAT: Copy of the clean model.
    # HOW: deepcopy.
    # WHY: The clean model stays unchanged.
    # PROS & CONS: + Safe. - Memory.
    # EXAMPLE: deepcopy(clean) trains a copy; the clean model's weights stay unchanged.
    model = copy.deepcopy(clean)
    # WHAT: Skip or train.
    # HOW: If finished, load the weights.
    # WHY: No repeated work.
    # PROS & CONS: + Saves time. - None.
    # EXAMPLE: completed checkpoint found.
    if os.path.exists(ckpt) and is_training_complete(hist_path, tag, args.epochs):
        load_checkpoint(model, ckpt, args.device)
        print(f"[{label}] completed checkpoint found -> skipping training")
    # WHAT: Runs adversarial training.
    # HOW: adversarial_finetune.
    # WHY: The defence.
    # PROS & CONS: + Resumable. - Time.
    # EXAMPLE: ResNet-18 about 1 hour.
    else:
        print(f"[{label}] adversarial fine-tuning: {args.epochs} epochs, pgd5, eps=8/255", flush=True)
        adversarial_finetune(model, loader, epochs=args.epochs, eps=EPS_TRAIN, method="pgd5", lr=1e-4,
                             device=args.device, tag=tag, ckpt_dir=args.ckpt_dir, drive_dir=args.drive_dir,
                             max_batches=args.max_batches)
    # WHAT: Returns the defended model.
    # HOW: eval().
    # WHY: Ready for evaluation.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: adv_stage returns the defended model in test mode, ready for evaluate_epsilon_sweep.
    return model.eval()


# WHAT: ResNet-20 trained from scratch, then adversarially fine-tuned.
# HOW: Random-weight architecture, 100 epochs of clean training, saved, then adv_stage.
# WHY: Satisfies 'train ResNet' fully.
# PROS & CONS: + Fully our own model. - 20-30 minutes of GPU time.
# EXAMPLE: Returns {'ResNet20-Scratch-Clean': ..., 'ResNet20-Scratch-AdvTrained': ...}.
def build_resnet20(args, loader, hist_path):
    """ResNet-20 trained from random initialisation, then adversarially fine-tuned."""
    # WHAT: Path of the clean model.
    # HOW: Folder plus file.
    # WHY: Reuse after a disconnect.
    # PROS & CONS: + Resumable. - None.
    # EXAMPLE: checkpoints_extra/resnet20s_clean.pth
    path = os.path.join(args.ckpt_dir, "resnet20s_clean.pth")
    # WHAT: Architecture only, no trained weights.
    # HOW: pretrained=False.
    # WHY: From-scratch training starts from random weights.
    # PROS & CONS: + Honest. - None.
    # EXAMPLE: 0.27M random weights.
    shell = load_cnn_clean(args.device, pretrained=False)  # architecture only, random weights, no download of trained weights
    # WHAT: Load or train.
    # HOW: Existing checkpoint is loaded; otherwise training runs and saves.
    # WHY: Saves time.
    # PROS & CONS: + Resumable. - None.
    # EXAMPLE: [ResNet20-Scratch-Clean] training from scratch for 100 epochs
    if os.path.exists(path):
        clean = load_checkpoint(shell, path, args.device)
        print("[ResNet20-Scratch-Clean] loaded from checkpoint")
    else:
        print(f"[ResNet20-Scratch-Clean] training from scratch for {args.scratch_epochs} epochs", flush=True)
        clean = train_from_scratch(shell, loader, args.scratch_epochs, args.device, args.max_batches)
        save_checkpoint(clean, path)
        sync_to_drive(args.ckpt_dir, args.drive_dir)
    # WHAT: Adversarial stage and return.
    # HOW: Both models in a dictionary.
    # WHY: Used by main().
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: 2 models.
    adv = adv_stage(clean, "resnet20s_adv", "ResNet20-Scratch-AdvTrained", args, loader, hist_path)
    return {"ResNet20-Scratch-Clean": clean, "ResNet20-Scratch-AdvTrained": adv}


# WHAT: ImageNet-pretrained ResNet-18: clean fine-tune, then adversarial fine-tuning.
# HOW: 224x224 input and about 11M weights, closer to ViT-Tiny than ResNet-20.
# WHY: A closer match in size and input than ResNet-20 for the comparison with the ViT.
# PROS & CONS: + Fairer comparison. - About 1 hour of GPU time.
# EXAMPLE: Returns two models.
def build_resnet18(args, loader, hist_path):
    """ImageNet-pretrained ResNet-18 (224x224 input): clean fine-tune, then adversarial fine-tuning."""
    # WHAT: Path of the clean ResNet-18.
    # HOW: Folder plus file.
    # WHY: Resume.
    # PROS & CONS: + Resumable. - None.
    # EXAMPLE: checkpoints_extra/resnet18_clean.pth
    path = os.path.join(args.ckpt_dir, "resnet18_clean.pth")
    # WHAT: Load or train.
    # HOW: Load if saved; otherwise download pretrained weights and fine-tune 5 epochs.
    # WHY: The new 10-class head must learn CIFAR-10.
    # PROS & CONS: + Resumable. - Time.
    # EXAMPLE: [ResNet18-Clean] clean fine-tune on CIFAR-10 for 5 epochs
    if os.path.exists(path):
        shell, _ = load_resnet18_clean(args.device, pretrained=False)
        clean = load_checkpoint(shell, path, args.device)
        print("[ResNet18-Clean] loaded from checkpoint")
    else:
        clean, _ = load_resnet18_clean(args.device, pretrained=True)
        print(f"[ResNet18-Clean] clean fine-tune on CIFAR-10 for {args.r18_clean_epochs} epochs", flush=True)
        clean_finetune(clean, loader, epochs=args.r18_clean_epochs, lr=2e-4, device=args.device,
                       max_batches=args.max_batches, desc="ResNet18-clean")
        save_checkpoint(clean, path)
        sync_to_drive(args.ckpt_dir, args.drive_dir)
    # WHAT: Adversarial stage and return.
    # HOW: Both models in a dictionary.
    # WHY: Used by main().
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: 2 models.
    adv = adv_stage(clean, "resnet18_adv", "ResNet18-AdvTrained", args, loader, hist_path)
    return {"ResNet18-Clean": clean, "ResNet18-AdvTrained": adv}


# WHAT: Runs the chosen experiments and evaluates them.
# HOW: Parse options, build models, sweep, strong attack, clean accuracy on all 10,000 test images, save CSV and
#      JSON.
# WHY: One command gives all numbers needed for the report.
# PROS & CONS: + Resumable, with a --quick smoke test. - Not yet run by us.
# EXAMPLE: python scripts/extra_cnn_experiments.py --only resnet20
def main():
    # WHAT: Options.
    # HOW: --only, --quick, --epochs, --scratch-epochs, --strong-eps, --drive-dir, ...
    # WHY: Control cost and parallel runs.
    # PROS & CONS: + Flexible. - Many options.
    # EXAMPLE: --only resnet18
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["resnet20", "resnet18", "both"], default="both")
    p.add_argument("--quick", action="store_true", help="tiny smoke test (a few batches, 1 epoch, 32 test images)")
    p.add_argument("--epochs", type=int, default=10, help="adversarial fine-tuning epochs")
    p.add_argument("--scratch-epochs", type=int, default=100, help="ResNet-20 from-scratch epochs")
    p.add_argument("--r18-clean-epochs", type=int, default=5)
    p.add_argument("--eval-samples", type=int, default=1000)
    p.add_argument("--strong-eps", nargs="+", type=float, default=[8.0], help="eps (/255) for the PGD-50 x5 check")
    p.add_argument("--strong-steps", type=int, default=50)
    p.add_argument("--strong-restarts", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-root", default="data")
    p.add_argument("--ckpt-dir", default="checkpoints_extra")
    p.add_argument("--out-dir", default="outputs_extra_cnn")
    p.add_argument("--drive-dir", default=None, help="e.g. /content/drive/MyDrive/extra_ckpts (Colab backup)")
    # WHAT: Reads the options and sets defaults.
    # HOW: Device and settings.
    # WHY: Used below.
    # PROS & CONS: + Standard. - None.
    # EXAMPLE: a.only == 'both'.
    a = p.parse_args()

    a.device = get_device()
    a.max_batches, pgd_steps = 0, 10
    # WHAT: Shrinks everything for the smoke test.
    # HOW: 1 epoch, few batches, 32 images, separate folders.
    # WHY: Verifies the script in a few minutes before the long run.
    # PROS & CONS: + Cheap check. - Numbers invalid.
    # EXAMPLE: checkpoints_extra_quick/
    if a.quick:  # tiny run just to check that the code works end to end
        a.epochs, a.scratch_epochs, a.r18_clean_epochs = 1, 1, 1
        a.eval_samples, a.batch_size, a.num_workers = 32, 16, 0
        a.max_batches, pgd_steps = 2, 2
        a.strong_steps, a.strong_restarts = 3, 1
        a.ckpt_dir, a.out_dir, a.drive_dir = "checkpoints_extra_quick", "outputs_extra_cnn_quick", None
    # WHAT: Creates folders, seeds randomness and prints the setup.
    # HOW: makedirs, set_seed, print.
    # WHY: Repeatability.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: Device: cuda | only=both | quick=False
    os.makedirs(a.ckpt_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    set_seed(SEED)
    print(f"Device: {a.device} | only={a.only} | quick={a.quick}", flush=True)
    if a.device.type == "cpu" and not a.quick:
        print("WARNING: no GPU found. Use a Colab T4 GPU (or --quick).")

    # WHAT: Starts the stopwatch and restores backups.
    # HOW: Drive restore, data download, training loader.
    # WHY: Resume after a disconnect.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: total_minutes in the JSON.
    t_start = time.time()
    restore_from_drive(a.ckpt_dir, a.drive_dir)
    prepare_data(a.data_root, a.drive_dir)
    loader = get_train_loader(a.batch_size, a.data_root, 64 if a.quick else 0, num_workers=a.num_workers)
    hist_path = os.path.join(a.ckpt_dir, "training_history.json")

    # WHAT: Builds the selected models.
    # HOW: Calls build_resnet20 and/or build_resnet18.
    # WHY: Evaluation needs all of them.
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: 2 or 4 models.
    models = {}
    if a.only in ("resnet20", "both"):
        models.update(build_resnet20(a, loader, hist_path))
    if a.only in ("resnet18", "both"):
        models.update(build_resnet18(a, loader, hist_path))
    # WHAT: Freezes the weights.
    # HOW: requires_grad_(False).
    # WHY: Attacks need gradients only for the image.
    # PROS & CONS: + Faster. - None.
    # EXAMPLE: After the loop every parameter has requires_grad == False.
    for m in models.values():  # attacks only need gradients with respect to the image
        m.eval()
        for prm in m.parameters():
            prm.requires_grad_(False)

    # WHAT: Fixed evaluation subset.
    # HOW: Same 1000 images as the report.
    # WHY: Comparable numbers.
    # PROS & CONS: + Fair. - None.
    # EXAMPLE: (1000, 3, 32, 32).
    X, Y = get_eval_tensors(a.eval_samples, a.data_root)
    print(f"Evaluation subset: {len(X)} stratified test images (seed {SEED})", flush=True)
    # WHAT: FGSM and PGD-10 curves.
    # HOW: Same function as the main experiment.
    # WHY: Same accuracy-versus-epsilon curves.
    # PROS & CONS: + Comparable. - Time.
    # EXAMPLE: CSV extra_cnn_sweep_<only>.csv.
    sweep = evaluate_epsilon_sweep(models, X, Y, a.device, pgd_steps=pgd_steps)
    sweep.to_csv(os.path.join(a.out_dir, f"extra_cnn_sweep_{a.only}.csv"), index=False)

    # WHAT: Strong attack on the defended models.
    # HOW: PGD-50 with 5 restarts.
    # WHY: Same check as strong_attack_check.py.
    # PROS & CONS: + Trustworthy. - Time.
    # EXAMPLE: CSV extra_cnn_strong_<only>.csv.
    strong_rows = []
    for name, model in models.items():
        if "AdvTrained" not in name:
            continue
        for e in a.strong_eps:
            eps = e / 255.0
            print(f"[strong] {name} eps={e:g}/255", flush=True)
            s = robust_accuracy(model, X, Y, a.device, eps, a.strong_steps, a.strong_restarts,
                                eps / 10, 100, SEED)
            strong_rows.append({"model": name, "eps_255": e, f"pgd{a.strong_steps}_{a.strong_restarts}restarts": s})
    strong_df = pd.DataFrame(strong_rows)
    strong_df.to_csv(os.path.join(a.out_dir, f"extra_cnn_strong_{a.only}.csv"), index=False)

    # WHAT: Clean accuracy on the full test set.
    # HOW: All 10,000 images.
    # WHY: More reliable clean numbers (standard error about 0.3 points).
    # PROS & CONS: + Precise. - Extra minute.
    # EXAMPLE: clean_acc_full_test_set in the JSON.
    X10k, Y10k = get_eval_tensors(10000 if not a.quick else 64, a.data_root)  # clean accuracy on the full test set
    clean_full = {k: accuracy(m, X10k, Y10k, a.device) for k, m in models.items()}
    # WHAT: Run summary.
    # HOW: Settings, accuracies, minutes.
    # WHY: Provenance.
    # PROS & CONS: + Traceable. - None.
    # EXAMPLE: extra_cnn_run_info_<only>.json
    info = {"only": a.only, "quick": a.quick, "device": str(a.device), "epochs_adv": a.epochs,
            "scratch_epochs": a.scratch_epochs, "r18_clean_epochs": a.r18_clean_epochs,
            "eval_samples": len(X), "clean_acc_eval_subset": {k: accuracy(m, X, Y, a.device) for k, m in models.items()},
            "clean_acc_full_test_set": clean_full, "total_minutes": round((time.time() - t_start) / 60, 2)}
    # WHAT: Writes and prints the summary tables.
    # HOW: JSON and console tables.
    # WHY: Easy to copy into the report.
    # PROS & CONS: + Visible. - None.
    # EXAMPLE: Done in X min.
    save_json(info, os.path.join(a.out_dir, f"extra_cnn_run_info_{a.only}.json"))

    print("\n=== PGD-10 / FGSM accuracy at eps = 0 and 8/255 (N=%d) ===" % len(X))
    print(sweep[sweep.eps_255.isin([0, 8])].to_string(index=False))
    print("\n=== Stronger attack ===")
    print(strong_df.to_string(index=False))
    print("\n=== Clean accuracy on all test images ===")
    print(clean_full)
    print(f"\nDone in {info['total_minutes']} min. Results in {a.out_dir}/")


# WHAT: Starts main() when the file is run.
# HOW: Python sets __name__.
# WHY: Importable without running.
# PROS & CONS: + Standard. - None.
# EXAMPLE: python scripts/extra_cnn_experiments.py
if __name__ == "__main__":
    main()
