"""CLI entry point: CNN vs ViT adversarial robustness on CIFAR-10.

  python main.py --quick-test            # CPU smoke test (64 test images, 1 tiny epoch), no GPU needed
  python main.py --full-run              # Colab/Kaggle GPU: 10-epoch adversarial fine-tuning + full evaluation
  python main.py --epochs 10 --eval-samples 1000 --auto-push

Checkpoints in checkpoints/ are reused automatically: nothing that already exists is retrained.
"""
import argparse
import copy
import json
import os
import time

from src.data import get_eval_tensors, get_train_loader, prepare_data
from src.defense import adversarial_finetune, clean_finetune, is_training_complete
from src.evaluate import accuracy, evaluate_epsilon_sweep, evaluate_transferability, export_results
from src.models import load_checkpoint, load_cnn_clean, load_vit_clean, save_checkpoint
from src.utils import (SEED, get_device, git_auto_push, load_json, restore_from_drive, save_json, set_seed,
                       sync_to_drive)
from src.visualize import make_all_plots


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick-test", action="store_true", help="CPU smoke test: 64 images, 1 short epoch")
    mode.add_argument("--full-run", action="store_true", help="Full GPU run (default when no mode is given)")
    p.add_argument("--epochs", type=int, default=10, help="adversarial fine-tuning epochs")
    p.add_argument("--eval-samples", type=int, default=1000, help="stratified test subset size")
    p.add_argument("--eps", type=float, default=8.0, help="training epsilon in /255 units")
    p.add_argument("--cnn-method", choices=["pgd5", "rs_fgsm"], default="pgd5")
    p.add_argument("--vit-method", choices=["pgd5", "rs_fgsm"], default="rs_fgsm",
                   help="rs_fgsm keeps ViT adversarial training inside the Colab T4 time budget")
    p.add_argument("--vit-backbone", choices=["timm-tiny", "hf-base"], default="timm-tiny")
    p.add_argument("--vit-clean-epochs", type=int, default=5, help="clean fine-tune epochs for timm ViT-Tiny")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-train-samples", type=int, default=0, help="0 = full 50k training set")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-root", default="data")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--drive-dir", default=None, help="e.g. /content/drive/MyDrive/adv_ckpts (Colab)")
    p.add_argument("--force-retrain", action="store_true", help="ignore existing adversarial checkpoints")
    p.add_argument("--auto-push", action="store_true", help="git commit+push after each stage / every 5 epochs")
    return p.parse_args()


def clear_stale_quick_test_outputs(out_dir):
    """Quick-test artifacts are committed to git; delete them so they can never pass for full-run results."""
    info = load_json(os.path.join(out_dir, "run_info.json"), {}) or {}
    if info.get("mode") != "quick-test":
        return
    for name in ("acc_vs_epsilon.png", "clean_vs_robust_tradeoff.png", "transferability_heatmap.png",
                 "adversarial_examples_grid.png", "results_summary.csv", "results_summary.md",
                 "transferability.csv", "run_info.json"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            os.remove(p)
    print("[outputs] removed stale quick-test artifacts")


def main():
    args = parse_args()
    quick = args.quick_test
    device = get_device()
    set_seed(SEED)

    pgd_steps, max_batches, tag = 10, 0, ""
    if quick:
        args.epochs, args.eval_samples, args.batch_size = 1, 64, 16
        args.max_train_samples, args.vit_clean_epochs, args.num_workers = 64, 1, 0
        args.drive_dir = None   # never mix throw-away smoke-test checkpoints into the real Drive cache
        pgd_steps, max_batches = 2, 2
        tag = "QUICK-TEST smoke run (tiny N, 1 epoch, 2 PGD steps) - not a result"
    args.ckpt_dir = args.ckpt_dir or ("checkpoints_quick_test" if quick else "checkpoints")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    eps_train = args.eps / 255.0

    print(f"Device: {device} | mode: {'quick-test' if quick else 'full-run'} | ckpt: {args.ckpt_dir}")
    if device.type == "cpu" and not quick:
        print("WARNING: full-run on CPU will be extremely slow. Use Colab/Kaggle GPU (or --quick-test).")

    restore_from_drive(args.ckpt_dir, args.drive_dir)
    if not quick:
        clear_stale_quick_test_outputs(args.out_dir)
    push = (lambda msg: git_auto_push(msg)) if args.auto_push else (lambda msg: None)
    t_start = time.time()

    # Stage 1: clean baselines (pretrained, no/little training).
    # Models are loaded BEFORE the dataset so a network/hub failure shows up in seconds, not after a
    # long CIFAR-10 download.
    cnn_path = os.path.join(args.ckpt_dir, "cnn_clean.pth")
    vit_path = os.path.join(args.ckpt_dir, "vit_clean.pth")

    if os.path.exists(cnn_path):
        cnn_clean = load_checkpoint(load_cnn_clean(device, pretrained=False), cnn_path, device)
        print("[CNN-Clean] loaded from checkpoint")
    else:
        cnn_clean = load_cnn_clean(device)
        save_checkpoint(cnn_clean, cnn_path)

    vit_needs_ft = False
    if os.path.exists(vit_path):
        shell, _ = load_vit_clean(device, args.vit_backbone, pretrained=False)
        vit_clean = load_checkpoint(shell, vit_path, device)
        print("[ViT-Clean] loaded from checkpoint")
    else:
        vit_clean, vit_needs_ft = load_vit_clean(device, args.vit_backbone)

    prepare_data(args.data_root, args.drive_dir)
    train_loader = get_train_loader(args.batch_size, args.data_root, args.max_train_samples,
                                    num_workers=args.num_workers)

    if vit_needs_ft:
        print(f"[ViT-Clean] clean fine-tune on CIFAR-10 for {args.vit_clean_epochs} epochs")
        clean_finetune(vit_clean, train_loader, epochs=args.vit_clean_epochs, device=device,
                       max_batches=max_batches)
        save_checkpoint(vit_clean, vit_path)
    sync_to_drive(args.ckpt_dir, args.drive_dir)
    push("chore(train): clean CNN/ViT baseline checkpoints")

    # Stage 2: adversarial fine-tuning (skipped if already completed)
    hist_path = os.path.join(args.ckpt_dir, "training_history.json")
    adv_models = {}
    for tag_name, clean, method, label in (("cnn_adv", cnn_clean, args.cnn_method, "CNN-AdvTrained"),
                                           ("vit_adv", vit_clean, args.vit_method, "ViT-AdvTrained")):
        ckpt = os.path.join(args.ckpt_dir, f"{tag_name}.pth")
        model = copy.deepcopy(clean)
        if (os.path.exists(ckpt) and not args.force_retrain
                and is_training_complete(hist_path, tag_name, args.epochs)):
            load_checkpoint(model, ckpt, device)
            print(f"[{label}] completed checkpoint found -> skipping training")
        else:
            print(f"[{label}] adversarial fine-tuning: {args.epochs} epochs, {method}, eps={args.eps}/255")
            if args.force_retrain:
                for f in (ckpt, os.path.join(args.ckpt_dir, f"{tag_name}_state.pt")):
                    if os.path.exists(f):
                        os.remove(f)
            adversarial_finetune(model, train_loader, epochs=args.epochs, eps=eps_train, method=method,
                                 lr=args.lr, device=device, tag=tag_name, ckpt_dir=args.ckpt_dir,
                                 drive_dir=args.drive_dir, auto_push_every=5 if args.auto_push else 0,
                                 max_batches=max_batches)
            push(f"chore(train): {label} {args.epochs}-epoch adversarial fine-tuning finished")
        adv_models[label] = model.eval()

    models = {"CNN-Clean": cnn_clean, "ViT-Clean": vit_clean, **adv_models}
    for m in models.values():
        for prm in m.parameters():
            prm.requires_grad_(False)   # attacks need input grads only

    # Stage 3: evaluation
    X, Y = get_eval_tensors(args.eval_samples, args.data_root)
    print(f"Evaluation subset: {len(X)} stratified test images (seed {SEED})")
    sweep = evaluate_epsilon_sweep(models, X, Y, device, pgd_steps=pgd_steps)
    transfer = evaluate_transferability(models, X, Y, device, eps=8 / 255, pgd_steps=pgd_steps)
    note = (f"Quick-test smoke run only (N={len(X)}, {pgd_steps} PGD steps): not valid results."
            if quick else f"N={len(X)} stratified CIFAR-10 test images, seed {SEED}, PGD-{pgd_steps}.")
    export_results(sweep, transfer, args.out_dir, note)

    # Stage 4: figures
    make_all_plots(sweep, transfer, models, X, Y, device, args.out_dir, pgd_steps=pgd_steps, tag=tag)

    info = {"mode": "quick-test" if quick else "full-run", "device": str(device), "epochs": args.epochs,
            "eval_samples": len(X), "pgd_eval_steps": pgd_steps, "cnn_method": args.cnn_method,
            "vit_method": args.vit_method, "vit_backbone": args.vit_backbone,
            "clean_acc_full_subset": {k: accuracy(m, X, Y, device) for k, m in models.items()},
            "total_minutes": round((time.time() - t_start) / 60, 2)}
    save_json(info, os.path.join(args.out_dir, "run_info.json"))
    print(json.dumps(info, indent=2))
    push("results: evaluation tables and plots")
    print(f"Done in {info['total_minutes']} min. Outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
