"""Fast adversarial fine-tuning (10 epochs, AMP) + optional short clean fine-tune for the timm ViT.

Per batch:
  1. model.eval(), build x_adv (PGD-5 or RS-FGSM) under autocast.
  2. model.train(), loss = 0.5 * CE(f(x), y) + 0.5 * CE(f(x_adv), y).
     Clean and adversarial halves go through ONE forward pass on the concatenated batch [x; x_adv].
     With equal-sized halves, the mean CE over the 2B samples is exactly the 50/50 loss above.
  3. GradScaler step.
Checkpoints, history JSON and a resumable training-state file are written after every epoch.
"""
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .attacks import pgd_attack, rs_fgsm_attack
from .models import save_checkpoint
from .utils import git_auto_push, load_json, save_json, sync_to_drive


def _autocast(device, enabled):
    return torch.autocast(device_type=device.type, enabled=bool(enabled) and device.type == "cuda")


def make_train_attack(method, eps):
    if method == "pgd5":
        return lambda m, x, y: pgd_attack(m, x, y, eps=eps, alpha=2 / 255, steps=5, random_start=True, amp=True)
    if method == "rs_fgsm":
        return lambda m, x, y: rs_fgsm_attack(m, x, y, eps=eps, alpha=1.25 * eps, amp=True)
    raise ValueError(f"method must be 'pgd5' or 'rs_fgsm', got {method!r}")


def update_history(history_path, tag, entry=None, **meta):
    hist = load_json(history_path, {}) or {}
    rec = hist.setdefault(tag, {"epochs": []})
    rec.update(meta)
    if entry is not None:
        rec["epochs"] = [e for e in rec["epochs"] if e["epoch"] != entry["epoch"]] + [entry]
    save_json(hist, history_path)


def is_training_complete(history_path, tag, epochs):
    rec = (load_json(history_path, {}) or {}).get(tag, {})
    return bool(rec.get("completed")) and len(rec.get("epochs", [])) >= epochs


def adversarial_finetune(model, train_loader, epochs=10, eps=8 / 255, method="pgd5", lr=1e-4,
                         device=None, tag="cnn_adv", ckpt_dir="checkpoints", drive_dir=None,
                         auto_push_every=0, push_fn=git_auto_push, max_batches=0):
    """Adversarially fine-tune `model` in place and return the history list.

    max_batches > 0 limits batches per epoch (smoke tests only).
    Resumes automatically from checkpoints/<tag>_state.pt if a previous run was interrupted.
    """
    device = device or next(model.parameters()).device
    model.to(device)
    attack = make_train_attack(method, eps)
    use_amp = device.type == "cuda"

    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pth")
    state_path = os.path.join(ckpt_dir, f"{tag}_state.pt")
    hist_path = os.path.join(ckpt_dir, "training_history.json")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=device)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        start_epoch = st["epoch"]
        print(f"[{tag}] Resuming from epoch {start_epoch}/{epochs}")

    update_history(hist_path, tag, completed=False, method=method, eps=eps, lr=lr, target_epochs=epochs,
                   amp=use_amp, device=str(device))

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tot_loss = tot_clean = tot_adv = n = 0
        pbar = tqdm(train_loader, desc=f"[{tag}] Epoch {epoch + 1}/{epochs}", leave=False)
        for b, (x, y) in enumerate(pbar):
            if max_batches and b >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            model.eval()
            with _autocast(device, use_amp):
                x_adv = attack(model, x, y)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                logits = model(torch.cat([x, x_adv], dim=0))
            logits = logits.float()
            bs = x.size(0)
            loss = 0.5 * F.cross_entropy(logits[:bs], y) + 0.5 * F.cross_entropy(logits[bs:], y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tot_loss += loss.item() * bs
            tot_clean += (logits[:bs].argmax(1) == y).sum().item()
            tot_adv += (logits[bs:].argmax(1) == y).sum().item()
            n += bs
            pbar.set_postfix(loss=f"{tot_loss / n:.3f}", clean=f"{100 * tot_clean / n:.1f}%",
                             robust=f"{100 * tot_adv / n:.1f}%")
        pbar.close()
        scheduler.step()
        model.eval()

        entry = {"epoch": epoch + 1, "train_loss": tot_loss / max(n, 1),
                 "train_clean_acc": 100 * tot_clean / max(n, 1), "train_robust_acc": 100 * tot_adv / max(n, 1),
                 "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        print(f"[{tag}] Epoch {entry['epoch']}/{epochs} | loss {entry['train_loss']:.4f} | "
              f"clean {entry['train_clean_acc']:.2f}% | robust {entry['train_robust_acc']:.2f}% | "
              f"{entry['seconds']:.0f}s")

        # checkpoint after EVERY epoch
        save_checkpoint(model, ckpt_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1}, state_path)
        update_history(hist_path, tag, entry=entry)
        sync_to_drive(ckpt_dir, drive_dir)
        if auto_push_every and (epoch + 1) % auto_push_every == 0 and (epoch + 1) < epochs:
            push_fn(f"chore(train): {tag} epoch {epoch + 1}/{epochs} checkpoint")

    update_history(hist_path, tag, completed=True)
    if os.path.exists(state_path):
        os.remove(state_path)  # training finished; the resume state is no longer needed
    sync_to_drive(ckpt_dir, drive_dir)
    model.eval()
    return load_json(hist_path)[tag]["epochs"]


def clean_finetune(model, train_loader, epochs=5, lr=2e-4, device=None, max_batches=0, desc="ViT-clean"):
    """Plain (non-adversarial) fine-tune; used once to adapt the ImageNet timm ViT-Tiny to CIFAR-10."""
    device = device or next(model.parameters()).device
    use_amp = device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for epoch in range(epochs):
        model.train()
        tot = correct = n = 0
        pbar = tqdm(train_loader, desc=f"[{desc}] Epoch {epoch + 1}/{epochs}", leave=False)
        for b, (x, y) in enumerate(pbar):
            if max_batches and b >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tot += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
            pbar.set_postfix(loss=f"{tot / n:.3f}", acc=f"{100 * correct / n:.1f}%")
        pbar.close()
        scheduler.step()
        print(f"[{desc}] Epoch {epoch + 1}/{epochs} | loss {tot / max(n, 1):.4f} | acc {100 * correct / max(n, 1):.2f}%")
    return model.eval()
