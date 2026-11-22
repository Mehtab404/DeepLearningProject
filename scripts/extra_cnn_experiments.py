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
import argparse
import copy
import os
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repository root, so "src" can be imported
sys.path.insert(0, HERE)  # this folder, so strong_attack_check can be imported

from src.data import get_eval_tensors, get_train_loader, prepare_data  # noqa: E402
from src.defense import _autocast, adversarial_finetune, clean_finetune, is_training_complete  # noqa: E402
from src.evaluate import accuracy, evaluate_epsilon_sweep  # noqa: E402
from src.models import load_checkpoint, load_cnn_clean, load_resnet18_clean, save_checkpoint  # noqa: E402
from src.utils import SEED, get_device, restore_from_drive, save_json, set_seed, sync_to_drive  # noqa: E402
from strong_attack_check import robust_accuracy  # noqa: E402

EPS_TRAIN = 8 / 255  # training budget, same as the main experiments


def train_from_scratch(model, loader, epochs, device, max_batches=0, lr=0.1):
    """Normal (clean) training of a randomly initialised model: SGD + one-cycle learning-rate schedule."""
    use_amp = device.type == "cuda"
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    per_epoch = min(len(loader), max_batches) if max_batches else len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * per_epoch, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for epoch in range(epochs):
        model.train()
        tot = correct = n = 0
        t0 = time.time()
        for b, (x, y) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"[scratch] epoch {epoch + 1}/{epochs} | loss {tot / max(n, 1):.4f} | "
                  f"train acc {100 * correct / max(n, 1):.2f}% | {time.time() - t0:.0f}s/epoch", flush=True)
    return model.eval()


def adv_stage(clean, tag, label, args, loader, hist_path):
    """Adversarial fine-tuning with the same recipe as the main experiments (skips if already finished)."""
    ckpt = os.path.join(args.ckpt_dir, f"{tag}.pth")
    model = copy.deepcopy(clean)
    if os.path.exists(ckpt) and is_training_complete(hist_path, tag, args.epochs):
        load_checkpoint(model, ckpt, args.device)
        print(f"[{label}] completed checkpoint found -> skipping training")
    else:
        print(f"[{label}] adversarial fine-tuning: {args.epochs} epochs, pgd5, eps=8/255", flush=True)
        adversarial_finetune(model, loader, epochs=args.epochs, eps=EPS_TRAIN, method="pgd5", lr=1e-4,
                             device=args.device, tag=tag, ckpt_dir=args.ckpt_dir, drive_dir=args.drive_dir,
                             max_batches=args.max_batches)
    return model.eval()


def build_resnet20(args, loader, hist_path):
    """ResNet-20 trained from random initialisation, then adversarially fine-tuned."""
    path = os.path.join(args.ckpt_dir, "resnet20s_clean.pth")
    shell = load_cnn_clean(args.device, pretrained=False)  # architecture only, random weights, no download of trained weights
    if os.path.exists(path):
        clean = load_checkpoint(shell, path, args.device)
        print("[ResNet20-Scratch-Clean] loaded from checkpoint")
    else:
        print(f"[ResNet20-Scratch-Clean] training from scratch for {args.scratch_epochs} epochs", flush=True)
        clean = train_from_scratch(shell, loader, args.scratch_epochs, args.device, args.max_batches)
        save_checkpoint(clean, path)
        sync_to_drive(args.ckpt_dir, args.drive_dir)
    adv = adv_stage(clean, "resnet20s_adv", "ResNet20-Scratch-AdvTrained", args, loader, hist_path)
    return {"ResNet20-Scratch-Clean": clean, "ResNet20-Scratch-AdvTrained": adv}


def build_resnet18(args, loader, hist_path):
    """ImageNet-pretrained ResNet-18 (224x224 input): clean fine-tune, then adversarial fine-tuning."""
    path = os.path.join(args.ckpt_dir, "resnet18_clean.pth")
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
    adv = adv_stage(clean, "resnet18_adv", "ResNet18-AdvTrained", args, loader, hist_path)
    return {"ResNet18-Clean": clean, "ResNet18-AdvTrained": adv}


def main():
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
    a = p.parse_args()

    a.device = get_device()
    a.max_batches, pgd_steps = 0, 10
    if a.quick:  # tiny run just to check that the code works end to end
        a.epochs, a.scratch_epochs, a.r18_clean_epochs = 1, 1, 1
        a.eval_samples, a.batch_size, a.num_workers = 32, 16, 0
        a.max_batches, pgd_steps = 2, 2
        a.strong_steps, a.strong_restarts = 3, 1
        a.ckpt_dir, a.out_dir, a.drive_dir = "checkpoints_extra_quick", "outputs_extra_cnn_quick", None
    os.makedirs(a.ckpt_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    set_seed(SEED)
    print(f"Device: {a.device} | only={a.only} | quick={a.quick}", flush=True)
    if a.device.type == "cpu" and not a.quick:
        print("WARNING: no GPU found. Use a Colab T4 GPU (or --quick).")

    t_start = time.time()
    restore_from_drive(a.ckpt_dir, a.drive_dir)
    prepare_data(a.data_root, a.drive_dir)
    loader = get_train_loader(a.batch_size, a.data_root, 64 if a.quick else 0, num_workers=a.num_workers)
    hist_path = os.path.join(a.ckpt_dir, "training_history.json")

    models = {}
    if a.only in ("resnet20", "both"):
        models.update(build_resnet20(a, loader, hist_path))
    if a.only in ("resnet18", "both"):
        models.update(build_resnet18(a, loader, hist_path))
    for m in models.values():  # attacks only need gradients with respect to the image
        m.eval()
        for prm in m.parameters():
            prm.requires_grad_(False)

    X, Y = get_eval_tensors(a.eval_samples, a.data_root)
    print(f"Evaluation subset: {len(X)} stratified test images (seed {SEED})", flush=True)
    sweep = evaluate_epsilon_sweep(models, X, Y, a.device, pgd_steps=pgd_steps)
    sweep.to_csv(os.path.join(a.out_dir, f"extra_cnn_sweep_{a.only}.csv"), index=False)

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

    X10k, Y10k = get_eval_tensors(10000 if not a.quick else 64, a.data_root)  # clean accuracy on the full test set
    clean_full = {k: accuracy(m, X10k, Y10k, a.device) for k, m in models.items()}
    info = {"only": a.only, "quick": a.quick, "device": str(a.device), "epochs_adv": a.epochs,
            "scratch_epochs": a.scratch_epochs, "r18_clean_epochs": a.r18_clean_epochs,
            "eval_samples": len(X), "clean_acc_eval_subset": {k: accuracy(m, X, Y, a.device) for k, m in models.items()},
            "clean_acc_full_test_set": clean_full, "total_minutes": round((time.time() - t_start) / 60, 2)}
    save_json(info, os.path.join(a.out_dir, f"extra_cnn_run_info_{a.only}.json"))

    print("\n=== PGD-10 / FGSM accuracy at eps = 0 and 8/255 (N=%d) ===" % len(X))
    print(sweep[sweep.eps_255.isin([0, 8])].to_string(index=False))
    print("\n=== Stronger attack ===")
    print(strong_df.to_string(index=False))
    print("\n=== Clean accuracy on all test images ===")
    print(clean_full)
    print(f"\nDone in {info['total_minutes']} min. Results in {a.out_dir}/")


if __name__ == "__main__":
    main()
