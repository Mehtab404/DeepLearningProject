"""Fast adversarial fine-tuning (10 epochs, AMP) + optional short clean fine-tune for the timm ViT.

Per batch:
  1. model.eval(), build x_adv (PGD-5 or RS-FGSM) under autocast.
  2. model.train(), loss = 0.5 * CE(f(x), y) + 0.5 * CE(f(x_adv), y).
     Clean and adversarial halves go through ONE forward pass on the concatenated batch [x; x_adv].
     With equal-sized halves, the mean CE over the 2B samples is exactly the 50/50 loss above.
  3. GradScaler step.
Checkpoints, history JSON and a resumable training-state file are written after every epoch.
"""
# ---------------------------------------------------------------------------------------------------
# BIG PICTURE: ADVERSARIAL TRAINING (the defence)
# Idea: if the model is trained on adversarial examples, it learns to classify them correctly.
# This is a min-max game (Madry et al., 2018):
#     inner MAX: the attack finds the worst-case image x_adv for the current weights
#     outer MIN: the optimizer updates the weights to get x_adv (and x) right
# Cost: robust models usually lose some clean accuracy. That is the "clean vs robust trade-off".
# ---------------------------------------------------------------------------------------------------
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm    # progress bar

# ===================================================================================================
# EASY-LANGUAGE SUMMARY OF THIS FILE (read this first)
# ---------------------------------------------------------------------------------------------------
# What is this file?   It is the DEFENCE. It makes a model harder to fool.
# How?                 "Adversarial training": while training, we ALSO show the model fooled pictures
#                      (made by the attacks in attacks.py) together with the correct label.
#                      Like a student who practises with the hardest, trickiest questions.
# Each batch of training does these steps:
#   1. Take a batch of normal pictures.
#   2. Make an attacked copy of each picture (PGD-5 or RS-FGSM).
#   3. Give BOTH (normal + attacked) to the model, and compute the loss on both.
#   4. Update the model's weights so it gets both right next time.
# Price of safety: the model usually becomes a bit LESS accurate on normal pictures.
#                  This is the "clean vs robust trade-off".
# Other things in this file:
#   - Checkpoint after every epoch, so if Colab crashes we can continue instead of starting again.
#   - clean_finetune(): normal training with no attack (used only to prepare the ViT).
# Words: epoch = one full pass through all training images; batch = a small group of images;
#        loss = how wrong the model is; optimizer = the thing that updates the weights.
# ===================================================================================================

from .attacks import pgd_attack, rs_fgsm_attack
from .models import save_checkpoint
from .utils import git_auto_push, load_json, save_json, sync_to_drive


def _autocast(device, enabled):
    # Mixed precision: on a GPU, most maths runs in fp16 (half precision), which is about 2x faster
    # and uses less memory. It is turned off on the CPU.
    return torch.autocast(device_type=device.type, enabled=bool(enabled) and device.type == "cuda")


def make_train_attack(method, eps):
    # Returns the attack used to create adversarial images DURING TRAINING.
    #   "pgd5":    PGD with 5 steps of size 2/255. Stronger, but 5 gradient computations per batch.
    #   "rs_fgsm": one step with a random start. About 2.5x faster, but can be less stable.
    # (Evaluation always uses the stronger PGD-10, see evaluate.py.)
    if method == "pgd5":
        return lambda m, x, y: pgd_attack(m, x, y, eps=eps, alpha=2 / 255, steps=5, random_start=True, amp=True)
    if method == "rs_fgsm":
        return lambda m, x, y: rs_fgsm_attack(m, x, y, eps=eps, alpha=1.25 * eps, amp=True)
    raise ValueError(f"method must be 'pgd5' or 'rs_fgsm', got {method!r}")


def update_history(history_path, tag, entry=None, **meta):
    # Keeps a JSON log of training (loss and accuracy for each epoch) in checkpoints/training_history.json.
    # `tag` is the model name (e.g. "cnn_adv"). `meta` holds settings such as method, eps and lr.
    hist = load_json(history_path, {}) or {}
    rec = hist.setdefault(tag, {"epochs": []})
    rec.update(meta)
    if entry is not None:
        # Replace an old entry for the same epoch (if the run was resumed) instead of adding a duplicate.
        rec["epochs"] = [e for e in rec["epochs"] if e["epoch"] != entry["epoch"]] + [entry]
    save_json(hist, history_path)


def is_training_complete(history_path, tag, epochs):
    # True if this model already finished all its epochs, so main.py can skip retraining it.
    rec = (load_json(history_path, {}) or {}).get(tag, {})
    return bool(rec.get("completed")) and len(rec.get("epochs", [])) >= epochs


def adversarial_finetune(model, train_loader, epochs=10, eps=8 / 255, method="pgd5", lr=1e-4,
                         device=None, tag="cnn_adv", ckpt_dir="checkpoints", drive_dir=None,
                         auto_push_every=0, push_fn=git_auto_push, max_batches=0):
    """Adversarially fine-tune `model` in place and return the history list.

    max_batches > 0 limits batches per epoch (smoke tests only).
    Resumes automatically from checkpoints/<tag>_state.pt if a previous run was interrupted.
    """
    # "Fine-tune" = start from the already-trained clean model and keep training it for a few epochs
    # (much cheaper than training from scratch).
    device = device or next(model.parameters()).device
    model.to(device)
    attack = make_train_attack(method, eps)
    use_amp = device.type == "cuda"

    # Files written by this function:
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pth")             # final weights (used for evaluation)
    state_path = os.path.join(ckpt_dir, f"{tag}_state.pt")       # full state to resume after a crash
    hist_path = os.path.join(ckpt_dir, "training_history.json")  # loss/accuracy log

    # AdamW: the Adam optimizer with weight decay (a small penalty on large weights, against overfitting).
    # A small lr (1e-4) is used because we only fine-tune an already good model.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine schedule: the learning rate starts at lr and smoothly decreases towards 0 by the last epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # GradScaler: with fp16, small gradients can round to 0. The scaler multiplies the loss by a large
    # number before backward() and divides the gradients again before the optimizer step.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Resume: if Colab disconnected in the middle of training, continue from the last finished epoch.
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
        # Running totals for this epoch: loss, correct clean predictions, correct adversarial predictions,
        # number of images seen.
        tot_loss = tot_clean = tot_adv = n = 0
        pbar = tqdm(train_loader, desc=f"[{tag}] Epoch {epoch + 1}/{epochs}", leave=False)
        for b, (x, y) in enumerate(pbar):
            if max_batches and b >= max_batches:
                break    # only used by --quick-test
            # Move the batch to the GPU. non_blocking=True lets the copy overlap with computation.
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # STEP 1 (inner max): create adversarial images for the current weights.
            # eval() while attacking so dropout/BatchNorm statistics are not disturbed by crafting x_adv.
            model.eval()
            with _autocast(device, use_amp):
                x_adv = attack(model, x, y)

            # STEP 2 (outer min): update the weights.
            model.train()    # back to training mode (BatchNorm updates statistics, dropout is on)
            optimizer.zero_grad(set_to_none=True)    # clear the gradients from the previous batch
            with _autocast(device, use_amp):
                # Clean and adversarial images go through the model together in one batch of size 2*bs.
                # One forward pass is faster than two separate ones.
                logits = model(torch.cat([x, x_adv], dim=0))
            logits = logits.float()
            bs = x.size(0)
            # logits[:bs] = predictions for clean images, logits[bs:] = predictions for adversarial images.
            # Mixed adversarial training: half the loss on clean images (keeps clean accuracy),
            # half on adversarial images (buys robustness). This is the clean/robust trade-off knob.
            loss = 0.5 * F.cross_entropy(logits[:bs], y) + 0.5 * F.cross_entropy(logits[bs:], y)
            # EASY: loss = half "mistake on normal pictures" + half "mistake on attacked pictures".
            #       Lower loss = model is right more often on both kinds.

            # Backpropagation with the GradScaler (see above), then one optimizer step.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EASY: backward() = work out how each weight should change; step() = actually change the weights.

            # Statistics for the progress bar. argmax(1) = index of the highest score = predicted class.
            # EASY: "correct" = the class with the highest score equals the true label.
            tot_loss += loss.item() * bs
            tot_clean += (logits[:bs].argmax(1) == y).sum().item()
            tot_adv += (logits[bs:].argmax(1) == y).sum().item()
            n += bs
            pbar.set_postfix(loss=f"{tot_loss / n:.3f}", clean=f"{100 * tot_clean / n:.1f}%",
                             robust=f"{100 * tot_adv / n:.1f}%")
        pbar.close()
        scheduler.step()    # lower the learning rate once per epoch (cosine schedule)
        model.eval()

        # Summary of this epoch. max(n, 1) avoids a division by zero.
        entry = {"epoch": epoch + 1, "train_loss": tot_loss / max(n, 1),
                 "train_clean_acc": 100 * tot_clean / max(n, 1), "train_robust_acc": 100 * tot_adv / max(n, 1),
                 "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        print(f"[{tag}] Epoch {entry['epoch']}/{epochs} | loss {entry['train_loss']:.4f} | "
              f"clean {entry['train_clean_acc']:.2f}% | robust {entry['train_robust_acc']:.2f}% | "
              f"{entry['seconds']:.0f}s")

        # checkpoint after EVERY epoch
        # .pth = weights only. _state.pt = weights + optimizer + scheduler + scaler + epoch number,
        # which is everything needed to continue training exactly where it stopped.
        save_checkpoint(model, ckpt_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1}, state_path)
        update_history(hist_path, tag, entry=entry)
        sync_to_drive(ckpt_dir, drive_dir)    # backup copy on Google Drive (Colab only)
        if auto_push_every and (epoch + 1) % auto_push_every == 0 and (epoch + 1) < epochs:
            push_fn(f"chore(train): {tag} epoch {epoch + 1}/{epochs} checkpoint")

    # Training finished: mark it as complete so it is not repeated next time.
    update_history(hist_path, tag, completed=True)
    if os.path.exists(state_path):
        os.remove(state_path)  # training finished; the resume state is no longer needed
    sync_to_drive(ckpt_dir, drive_dir)
    model.eval()
    return load_json(hist_path)[tag]["epochs"]


def clean_finetune(model, train_loader, epochs=5, lr=2e-4, device=None, max_batches=0, desc="ViT-clean"):
    """Plain (non-adversarial) fine-tune; used once to adapt the ImageNet timm ViT-Tiny to CIFAR-10."""
    # The timm ViT was pretrained on ImageNet and got a brand-new 10-class head (see models.py).
    # This normal training loop (no attacks) teaches it CIFAR-10 first, so we get a fair "ViT-Clean" baseline.
    # It is the standard PyTorch loop: forward -> loss -> backward -> optimizer step.
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
                logits = model(x)                          # forward pass on CLEAN images only
            loss = F.cross_entropy(logits.float(), y)      # how wrong the predictions are
            scaler.scale(loss).backward()                  # compute gradients for the weights
            scaler.step(optimizer)                         # update the weights
            scaler.update()
            tot += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
            pbar.set_postfix(loss=f"{tot / n:.3f}", acc=f"{100 * correct / n:.1f}%")
        pbar.close()
        scheduler.step()
        print(f"[{desc}] Epoch {epoch + 1}/{epochs} | loss {tot / max(n, 1):.4f} | acc {100 * correct / max(n, 1):.2f}%")
    return model.eval()
